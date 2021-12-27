# TVM源代码解读

在`driver_api.cc`中把写的TE表达式lower成IRModule是由一下的函数完成的：

```C++

IRModule LowerSchedule(te::Schedule sch, const Array<ObjectRef>& args, const std::string& name,
                       const std::unordered_map<te::Tensor, tir::Buffer>& binds, bool simple_mode) {
  IRModule mod = ScheduleToModule(std::move(sch), args, name, binds);
  // Get the legacy TE pass list
  Array<transform::Pass> pass_list = CreatePassList(simple_mode);
  return LowerWithPassList(mod, pass_list);
}

```
可以看到由两部分组成：
第一部分是把`Schedule`转成`IRModule`（也就是由`TIR`组成的），
第二部分就是一系列的`transform`的pass。
先看是如何到`IRModule`的部分，改的重点有可能是后续中加pass。

```C++
IRModule ScheduleToModule(te::Schedule sch, const Array<ObjectRef>& args, const std::string& name,
                          const std::unordered_map<te::Tensor, tir::Buffer>& binds) {
  // Convert te schedule to IRModule
  Array<ObjectRef> out_arg_list;
  transform::PassContext pass_ctx = transform::PassContext::Current();

  sch = sch.normalize();

  // Before TIR transformation.
  Map<tir::IterVar, Range> bounds = te::InferBound(sch);
  tir::Stmt stmt = te::ScheduleOps(sch, std::move(bounds), false);
  bool compact = te::VerifyCompactBuffer(stmt);

  Map<te::Tensor, tir::Buffer> out_binds;
  GetBinds(args, compact, binds, &out_binds, &out_arg_list);

  // Build the function
  // At this point binds is only te::Tensors
  tir::PrimFunc f = te::SchedulePostProcToPrimFunc(out_arg_list, std::move(stmt), out_binds);
  f = WithAttr(std::move(f), "global_symbol", runtime::String(name));

  // Mark this schedule as being converted from an TE schedule. Makes sure that
  // the correct TE passes are run.
  f = WithAttr(std::move(f), "from_legacy_te_schedule", Bool(true));

  bool noalias = pass_ctx->GetConfig<Bool>("tir.noalias", Bool(true)).value();

  if (noalias) {
    f = WithAttr(std::move(f), "tir.noalias", Bool(true));
  }
  return IRModule(Map<GlobalVar, BaseFunc>({{GlobalVar(name), f}}));
}
```

第一个先看`sch.normalize()`，原型定义在`include/tvm/te/schedle.h`中，
实现在`te/schedule/schedule_dataflow_rewrite.cc`中。
inline op（也就是fusion）是发生在`InjectInline`中的。
在schedule的时候，如果调用了`sch[op].compute_inline()`，
则这个op的`staged->attach_type=kInline`，在`InjectInline`中正好对应。
看看TVM是如何做垂直fusion的。
其限制fusion的对象只能有一个output。
做inline分两个部分，第一部分直接把被inline的那For循环的body直接替换过去，
第二步是更新dateflow。我们在其中增加了检查inline的对象的axis是否完全一致的代码。



### TVM中循环长度为1的优化

```python
def test_te_to_tir(n):
    A = te.placeholder((n,), dtype='int32', name="A")
    B = te.compute((n,), lambda i : A[i]+1, name="compute_B")
    C = te.compute((n,), lambda j : B[j]+1, name="compute_C")
    sch = te.create_schedule(C.op)
    sch[B].compute_inline()
    mod = tvm.lower(sch, [A, B, C], simple_mode=True)
    print(mod)

if __name__=="__main__":
    test_te_to_tir(1)
```
如上图所示的Python代码，其运行的结果如下：
```rust
primfn(A_1: handle, compute_B_1: handle, compute_C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {compute_C: Buffer(compute_C_2: Pointer(int32), int32, [1], []),
             A: Buffer(A_2: Pointer(int32), int32, [1], []),
             compute_B: Buffer(compute_B_2: Pointer(int32), int32, [1], [])}
  buffer_map = {A_1: A, compute_B_1: compute_B, compute_C_1: compute_C} {
  compute_C_2[0] = ((int32*)A_2[0] + 2)
}
```
可以看到for循环就被优化掉了。这个过程发生在`te::InferBound`的过程中。
在`src/te/schedule/bound.cc`中，
`InferRootBound(stage, ctx, &ret);`根据output的`IterVar`的`dom`来判断，
之后会有一个Analyzer的对PrimExpr进行分析化简，如果其迭代长度为1，就会给这个var存为1，否则就保存其range。
```C++
void Analyzer::Bind(const Var& var, const Range& range, bool allow_override) {
  ICHECK(range.defined());
  if (tir::is_one(range->extent)) {
    this->Bind(var, range->min, allow_override);
  } else {
    this->const_int_bound.Bind(var, range, allow_override);
  }
  // skip modular_set
  // skip rewrite simplify
}
```

ScheduleOps函数比较重要，这个也要好好看一下。
好像是在schedule.normalize的时候处理了。那就再看看normalize的地方。



### TVM中computeOp IterVar的bound是如何推断出来的
在`python/tvm/te/operation.py`中, `shape`应该是output的shape，
然后lambda中的变量其迭代范围就是`shape`中的大小，核心这句话：
`dim_var = [tvm.tir.IterVar((0, s), x, 0) for x, s in zip(arg_names, shape[:out_ndim])]`。所以是在Python中构造的这些数据结构。
```Python
def compute(shape, fcompute, name="compute", tag="", attrs=None):
    shape = (shape,) if isinstance(shape, tvm.tir.PrimExpr) else shape
    # for python3
    shape = tuple([int(s) if isinstance(s, float) else s for s in shape])
    ndim = len(shape)
    code = fcompute.__code__

    out_ndim = ndim
    if code.co_argcount == 0:
        arg_names = ["i%d" % i for i in range(ndim)]
    else:
        arg_names = code.co_varnames[: code.co_argcount]
        out_ndim = code.co_argcount

    if out_ndim != len(arg_names):
        raise ValueError("fcompute do not match dimension, ndim=%d" % ndim)

    dim_var = [tvm.tir.IterVar((0, s), x, 0) for x, s in zip(arg_names, shape[:out_ndim])]
    body = fcompute(*[v.var for v in dim_var])

    if isinstance(body, _tensor.TensorIntrinCall):
        for i, s in enumerate(shape[out_ndim:]):
            var_name = "ax" + str(i)
            dim_var.append(tvm.tir.IterVar((0, s), var_name, 4))
        op_node = _ffi_api.TensorComputeOp(
            name,
            tag,
            dim_var,
            body.reduce_axis,
            out_ndim,
            body.intrin,
            body.tensors,
            body.regions,
            body.scalar_inputs,
        )
    else:
        if not isinstance(body, (list, tuple)):
            body = [body]
        body = convert(body)
        op_node = _ffi_api.ComputeOp(name, tag, attrs, dim_var, body)

    num = op_node.num_outputs
    outputs = tuple(op_node.output(i) for i in range(num))
    return outputs[0] if num == 1 else outputs
```


### 关于relay的op
relay底层的实现也是用te来做的。以`relay.add`为例，其实现为`_make.add(lhs, rhs)`,
具体实现在`src/relay/op/tensor/binary.cc`,宏展开后
```C++
TVM_REGISTER_GLOBAL("relay.op._make." "add").set_body_typed([](Expr lhs, Expr rhs) { static const Op& op = Op::Get("add"); return Call(op, {lhs, rhs}, Attrs(), {}); }); RELAY_REGISTER_OP("add") .set_num_inputs(2) .add_argument("lhs", "Tensor", "The left hand side tensor.") .add_argument("rhs", "Tensor", "The right hand side tensor.") .add_type_rel("Broadcast", BroadcastRel) .set_attr<TOpPattern>("TOpPattern", kBroadcast) .set_attr<TOpIsStateful>("TOpIsStateful", false) .set_attr<FInferCorrectLayout>("FInferCorrectLayout", BinaryBroadcastLayout)
```
其调用的为`topi::add`，定义在`include/tvm/topi/broadcast.h`中：宏展开如下：
```C++
inline tvm::PrimExpr add(const tvm::PrimExpr &a, const tvm::PrimExpr &b) {
  { return a + b; };
}
inline tvm::te::Tensor add(const tvm::te::Tensor &A, const tvm::te::Tensor &B,
                           std::string name = "T_"
                                              "add",
                           std::string tag = kBroadcast) {
  auto l = [](tvm::PrimExpr a, tvm::PrimExpr b) {
    { return a + b; };
  };
  return detail::WithBroadcast(l, A, B, name, tag);
}
inline tvm::te::Tensor add(const tvm::te::Tensor &A, const tvm::PrimExpr &B,
                           std::string name = "T_"
                                              "add",
                           std::string tag = kElementWise) {
  auto l = [](tvm::PrimExpr a, tvm::PrimExpr b) {
    { return a + b; };
  };
  return tvm::te::compute(
      A->shape,
      [&](const ::tvm::Array<::tvm::tir::Var> &i) { return l(A(i), B); }, name,
      tag);
}
inline tvm::te::Tensor add(const tvm::PrimExpr &A, const tvm::te::Tensor &B,
                           std::string name = "T_"
                                              "add",
                           std::string tag = kElementWise) {
  auto l = [&](tvm::PrimExpr a, tvm::PrimExpr b) {
    { return a + b; };
  };
  return tvm::te::compute(
      B->shape,
      [&](const ::tvm::Array<::tvm::tir::Var> &i) { return l(A, B(i)); }, name,
      tag);
}

```
所以底层实现仍然是te，而且一个op也会由多个te.compute来组成，例如softmax函数（在`include/tvm/topi/nn/softmax.h`中），
由te和tir的来完成。

### TVM中的fuseOps pass
tvm中fusion的例子见`tests/python/relay/test_pass_fuse_ops.py`。
需要搞明白一个问题`f1 = f1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))`中`Primitive`的含义。
fusion的pass在`src/relay/transforms/fuse_ops.cc`中，
其流程是把图切分，构建post-dominate tree，然后分子图进行融合。
首先是构造DAG图，确实是一个递归的过程，`class IndexedForwardGraph::Creator : private ExprVisitor`,其实现了各种`VisitExpr_`函数。例如访问`FunctionNode`，Update函数只是构建了一个`IndexedForwardGraph::Node`节点以及设置一下他们和父节点的边`IndexedForwardGraph::Edge`，并没有往`graph_.post_dfs_oder`里面加节点。
```C++
  // Post order tree
  void VisitExpr_(const FunctionNode* op) final {
    // Skip the function that should be handled by external codegen.
    if (op->GetAttr<String>(attr::kCompiler).defined()) return;

    for (auto param : op->params) {
      this->Update(param, nullptr, kOpaque);
    }
    this->Update(op->body, nullptr, kOpaque);
    ExprVisitor::VisitExpr_(op);
  }
```
而什么情况下才会加节点呢？看调用`this->AddNode`的。
例如对于`VarNode`就是直接添加进去：
```C++
void VisitExpr_(const VarNode* op) final { this->AddNode(op); }
```
而如果是`IfNode`，可以看到先给cond，然后是true_branch和false_branch创建节点，
之后才给op更新
```C++
  void VisitExpr_(const IfNode* op) final {
    // do not fuse through if.
    this->Update(op->cond, nullptr, kOpaque);
    this->Update(op->true_branch, nullptr, kOpaque);
    this->Update(op->false_branch, nullptr, kOpaque);
    ExprVisitor::VisitExpr_(op);
    this->AddNode(op);
  }
```
其中`ExprVisitor::VisitExpr_(op);`这句话中调用下面的函数，分别访问了其每个成员，所以就是一个递归的（并且是dfs的过程）构建DAG图。
```C++
void ExprVisitor::VisitExpr_(const IfNode* op) {
  this->VisitSpan(op->span);
  this->VisitExpr(op->cond);
  this->VisitExpr(op->true_branch);
  this->VisitExpr(op->false_branch);
}
```
统计一下调用`AddNode`的Relay IR，发现有以下：
```
ConstantNode
CallNode
TupleNode
TupleGetItemNode
VarNode
LetNode
IfNode
RefCreateNode
RefReadNode
RefWriteNode
MatchNode
```

然后看Update（也就是构造父子关系的），发现只有如下情况会有父子关系：
```C++
CallNode中和call->args
TupleNode中和tuple->fields
TupleGetItemNode中只有所有tuple_type中都是tensor，和op->tuple
```
其他的都是父亲为null。
。需要gdb来展示调用的trace。

### relay op到relay IR的表示
通过如下的例子展示一下relay的op到relay，可以看到add这样的op都是通过
```Python
def before():
    x = relay.var("data", shape=(10, 20))
    y = relay.add(x, relay.const(1, "float32"))
    z = relay.exp(y)
    w = relay.squeeze(z)
    return relay.Function([x], w)
```

```Rust
node[0], Var(data, ty=TensorType([10, 20], float32)) outputs=[2, ]
node[1], Constant(1.0) outputs=[2, ]
node[2], CallNode(Op(add), [Var(data, ty=TensorType([10, 20], float32)), Constant(1.0)], (nullptr), [TensorType([10, 20], float32), TensorType([], float32)]) outputs=[3, ]
node[3], CallNode(Op(exp), [CallNode(Op(add), [Var(data, ty=TensorType([10, 20], float32)), Constant(1.0)], (nullptr), [TensorType([10, 20], float32), TensorType([], float32)])], (nullptr), [TensorType([10, 20], float32)]) outputs=[4, ]
node[4], CallNode(Op(squeeze), [CallNode(Op(exp), [CallNode(Op(add), [Var(data, ty=TensorType([10, 20], float32)), Constant(1.0)], (nullptr), [TensorType([10, 20], float32), TensorType([], float32)])], (nullptr), [TensorType([10, 20], float32)])], relay.attrs.SqueezeAttrs(0x55b47454e3d8), [TensorType([10, 20], float32)]) outputs=[]
```
在post-dominate-tree中，一个节点的post-dominate是其所有output的最小公共祖先（说明之前的猜想是正确的，反过来说，`A` dominate `B`就说明`B`的所有input都是`A`的子节点的output（包含`A`本身）。反过来难以理解，正向就好理解了。


### relay op的pattern都是什么以及是怎么设置的
例如在`src/relay/op/tensor/unary.cc`中，可以看到首先，log这个op在relay中被搞成了`Call`,其op就是tir中的log，然后其输入就是这个args，属性被设置为了`kElemWise`。
```C++
TVM_REGISTER_GLOBAL("relay.op._make." "log").set_body_typed(
    [](Expr data) { static const Op& op = Op::Get("log"); 
    return Call(op, {data}, Attrs(), {}); }); 
RELAY_REGISTER_OP("log") 
    .set_num_inputs(1) 
    .add_argument("data", "Tensor", "The input tensor.") 
    .add_type_rel("Identity", IdentityRel) 
    .set_attr<TOpPattern>("TOpPattern", kElemWise)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout)
```


```C++
Call::Call(Expr op, Array<Expr> args, Attrs attrs, Array<Type> type_args, Span span) {
  ObjectPtr<CallNode> n = make_object<CallNode>();
  n->op = std::move(op);
  n->args = std::move(args);
  n->attrs = std::move(attrs);
  n->type_args = std::move(type_args);
  n->span = std::move(span);
  n->saved_deleter_ = n->deleter_;
  n->deleter_ = CallNode::Deleter_;
  data_ = std::move(n);
}
```


### relay op的继承关系

有以下几种op，均继承自`Expr`
```C++
using Expr = tvm::RelayExpr;
using ExprNode = tvm::RelayExprNode;
using BaseFunc = tvm::BaseFunc;
using BaseFuncNode = tvm::BaseFuncNode;
using GlobalVar = tvm::GlobalVar;
using GlobalVarNode = tvm::GlobalVarNode;

class ConstantNode : public ExprNode {
class Constant : public Expr {
class TupleNode : public ExprNode {
class Tuple : public Expr {
class VarNode : public ExprNode {
class Var : public Expr {
class CallNode : public ExprNode {
class Call : public Expr {
class LetNode : public ExprNode {
class Let : public Expr {
class IfNode : public ExprNode {
class If : public Expr {
class TupleGetItemNode : public ExprNode {
class TupleGetItem : public Expr {
class RefCreateNode : public ExprNode {
class RefCreate : public Expr {
class RefReadNode : public ExprNode {
class RefRead : public Expr {
class RefWriteNode : public ExprNode {
class RefWrite : public Expr {
class TempExprNode : public ExprNode {
class TempExpr : public Expr {
```


### TVM中op的类型


```C++
RELAY_REGISTER_UNARY_OP(OpName)
RELAY_REGISTER_BINARY_OP(OpName)
RELAY_REGISTER_CMP_OP(OpName)
```
下面这些都是`UNARY`类型的，都是`kElemWise`

```C++
op_common.h:52:#define RELAY_REGISTER_UNARY_OP(OpName)                                        \
tensor/unary.cc:41:RELAY_REGISTER_UNARY_OP("log")
tensor/unary.cc:51:RELAY_REGISTER_UNARY_OP("log2")
tensor/unary.cc:61:RELAY_REGISTER_UNARY_OP("log10")
tensor/unary.cc:71:RELAY_REGISTER_UNARY_OP("tan")
tensor/unary.cc:81:RELAY_REGISTER_UNARY_OP("cos")
tensor/unary.cc:91:RELAY_REGISTER_UNARY_OP("cosh")
tensor/unary.cc:101:RELAY_REGISTER_UNARY_OP("sin")
tensor/unary.cc:111:RELAY_REGISTER_UNARY_OP("sinh")
tensor/unary.cc:121:RELAY_REGISTER_UNARY_OP("acos")
tensor/unary.cc:131:RELAY_REGISTER_UNARY_OP("acosh")
tensor/unary.cc:141:RELAY_REGISTER_UNARY_OP("asin")
tensor/unary.cc:151:RELAY_REGISTER_UNARY_OP("asinh")
tensor/unary.cc:161:RELAY_REGISTER_UNARY_OP("atan")
tensor/unary.cc:171:RELAY_REGISTER_UNARY_OP("atanh")
tensor/unary.cc:181:RELAY_REGISTER_UNARY_OP("exp")
tensor/unary.cc:191:RELAY_REGISTER_UNARY_OP("fast_exp")
tensor/unary.cc:201:RELAY_REGISTER_UNARY_OP("erf")
tensor/unary.cc:211:RELAY_REGISTER_UNARY_OP("fast_erf")
tensor/unary.cc:221:RELAY_REGISTER_UNARY_OP("sqrt")
tensor/unary.cc:231:RELAY_REGISTER_UNARY_OP("rsqrt")
tensor/unary.cc:241:RELAY_REGISTER_UNARY_OP("zeros_like")
tensor/unary.cc:246:RELAY_REGISTER_UNARY_OP("ones_like")
tensor/unary.cc:251:RELAY_REGISTER_UNARY_OP("sigmoid")
tensor/unary.cc:261:RELAY_REGISTER_UNARY_OP("copy")
tensor/unary.cc:316:RELAY_REGISTER_UNARY_OP("floor")
tensor/unary.cc:322:RELAY_REGISTER_UNARY_OP("ceil")
tensor/unary.cc:332:RELAY_REGISTER_UNARY_OP("trunc")
tensor/unary.cc:342:RELAY_REGISTER_UNARY_OP("round")
tensor/unary.cc:352:RELAY_REGISTER_UNARY_OP("sign")
tensor/unary.cc:362:RELAY_REGISTER_UNARY_OP("abs")
tensor/unary.cc:372:RELAY_REGISTER_UNARY_OP("tanh")
tensor/unary.cc:382:RELAY_REGISTER_UNARY_OP("fast_tanh")
tensor/unary.cc:392:RELAY_REGISTER_UNARY_OP("negative")
tensor/unary.cc:402:RELAY_REGISTER_UNARY_OP("logical_not")
tensor/unary.cc:412:RELAY_REGISTER_UNARY_OP("bitwise_not")
tensor/unary.cc:503:RELAY_REGISTER_UNARY_OP("isnan")
tensor/unary.cc:512:RELAY_REGISTER_UNARY_OP("isfinite")
tensor/unary.cc:521:RELAY_REGISTER_UNARY_OP("isinf")
```

下面这些都是`BINARY`类型的，都是`kBroadcast`
```C++
op_common.h:75:#define RELAY_REGISTER_BINARY_OP(OpName)                                                \
tensor/binary.cc:42:RELAY_REGISTER_BINARY_OP("add")
tensor/binary.cc:48:RELAY_REGISTER_BINARY_OP("subtract")
tensor/binary.cc:54:RELAY_REGISTER_BINARY_OP("right_shift")
tensor/binary.cc:59:RELAY_REGISTER_BINARY_OP("left_shift")
tensor/binary.cc:64:RELAY_REGISTER_BINARY_OP("maximum")
tensor/binary.cc:69:RELAY_REGISTER_BINARY_OP("minimum")
tensor/binary.cc:74:RELAY_REGISTER_BINARY_OP("divide")
tensor/binary.cc:79:RELAY_REGISTER_BINARY_OP("floor_divide")
tensor/binary.cc:84:RELAY_REGISTER_BINARY_OP("multiply")
tensor/binary.cc:89:RELAY_REGISTER_BINARY_OP("power")
tensor/binary.cc:94:RELAY_REGISTER_BINARY_OP("mod")
tensor/binary.cc:99:RELAY_REGISTER_BINARY_OP("floor_mod")
tensor/binary.cc:104:RELAY_REGISTER_BINARY_OP("logical_and")
tensor/binary.cc:109:RELAY_REGISTER_BINARY_OP("logical_or")
tensor/binary.cc:114:RELAY_REGISTER_BINARY_OP("logical_xor")
tensor/binary.cc:119:RELAY_REGISTER_BINARY_OP("bitwise_and")
tensor/binary.cc:124:RELAY_REGISTER_BINARY_OP("bitwise_or")
tensor/binary.cc:129:RELAY_REGISTER_BINARY_OP("bitwise_xor")
```

```C++
op_common.h:90:#define RELAY_REGISTER_CMP_OP(OpName)                                                   \
tensor/binary.cc:134:RELAY_REGISTER_CMP_OP("equal")
tensor/binary.cc:139:RELAY_REGISTER_CMP_OP("not_equal")
tensor/binary.cc:144:RELAY_REGISTER_CMP_OP("less")
tensor/binary.cc:149:RELAY_REGISTER_CMP_OP("less_equal")
tensor/binary.cc:154:RELAY_REGISTER_CMP_OP("greater")
tensor/binary.cc:159:RELAY_REGISTER_CMP_OP("greater_equal")
```

所有的`src/relay/op/anntation/annotation.cc`中所有的都是`kOpaque`;

`resize`为`kInjective`

`pad` `kInjective`

`upsampling` `kInjective`

transform中：

1. `shape` `kInjective`
2. `tile` `kInjective`
3. `broadcast_to` `kBroadcast`
4. `one_hot` `kOutEWiseFusable`
5. `full` `kElemWise`
6. `strided_slice` `kInjective`
7. `sparse_to_dense` `kOpaque`
8. `expand_dims` `kInjective`
9. `squeeze` `kInjective`

image中：

1. `resize` `kInjective`

memory:
1. `device_copy` `kOpaque`
2. `allocate, kill` `kOpaque`

pad：
1. `pad, mirror_pad` `kOpaque`

upsampling:
1. `upsampling` `kInjective`

einsum:
1. `einsum` `kInjective`

reduce类的
1. `argmax, argmin, max, min, sum, all, any, prod, mean, variance` `kCommReduce`


transform类很多，有的kElemWise，有的kBrodcast，用到哪个再查吧。