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
### 关于te::compute返回的ComputeOp
在如下的例子中，这个compute函数根据shape构造出`IterVar`，
之后调用传进来的fcompute函数执行args，也就是IterVar->var，这时候他是一个`Array<PrimExpr> body`

```C++
Tensor compute(Array<PrimExpr> shape, FCompute fcompute, std::string name, std::string tag,
               Map<String, ObjectRef> attrs) {
  // compute dimension.
  size_t ndim = shape.size();
  std::vector<IterVar> axis;
  std::vector<Var> args;
  for (size_t i = 0; i < ndim; ++i) {
    std::ostringstream os;
    os << "ax" << i;
    axis.emplace_back(IterVar(Range(0, shape[i]), Var(os.str(), shape[i].dtype()), kDataPar));
    args.push_back(axis.back()->var);
  }

  return ComputeOp(name, tag, attrs, axis, {fcompute(args)}).output(0);
}
```
然后我们可以看到根据上面传进来的axis的信息就构造出来了一个基本的`ComputeOp`，
其对应的filed都被设置了。然后通过VerifyComputeOp可以看到如何比较两个`tir::ReduceNode`是否相同。
```C++
ComputeOp::ComputeOp(std::string name, std::string tag, Map<String, ObjectRef> attrs,
                     Array<IterVar> axis, Array<PrimExpr> body) {
  if (!attrs.defined()) {
    attrs = Map<String, ObjectRef>();
  }
  auto n = make_object<ComputeOpNode>();
  n->name = std::move(name);
  n->tag = std::move(tag);
  n->attrs = std::move(attrs);
  n->axis = std::move(axis);
  n->body = std::move(body);
  if (n->body[0]->IsInstance<tir::ReduceNode>()) {
    const tir::ReduceNode* reduce = n->body[0].as<tir::ReduceNode>();
    n->reduce_axis = reduce->axis;
  }
  VerifyComputeOp(n.get());
  data_ = std::move(n);
}
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

我们跟踪`te::compute()`，可以看到
```C++
Tensor compute(Array<PrimExpr> shape, FCompute fcompute, std::string name, std::string tag,
               Map<String, ObjectRef> attrs) {
  // compute dimension.
  size_t ndim = shape.size();
  std::vector<IterVar> axis;
  std::vector<Var> args;
  for (size_t i = 0; i < ndim; ++i) {
    std::ostringstream os;
    os << "ax" << i;
    axis.emplace_back(IterVar(Range(0, shape[i]), Var(os.str(), shape[i].dtype()), kDataPar));
    args.push_back(axis.back()->var);
  }

  return ComputeOp(name, tag, attrs, axis, {fcompute(args)}).output(0);
}

```

### TVM中的fuseOps pass
tvm中fusion的例子见`tests/python/relay/test_pass_fuse_ops.py`。
需要搞明白一个问题`f1 = f1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))`中`Primitive`的含义。
fusion的pass在`src/relay/transforms/fuse_ops.cc`中，
其流程是把图切分，构建post-dominate tree，然后分子图进行融合。
首先是构造DAG图，确实是一个递归的过程，`class IndexedForwardGraph::Creator : private ExprVisitor`,其实现了各种`VisitExpr_`函数。例如访问`FunctionNode`，Update函数只是构建了一个`IndexedForwardGraph::Node`节点以及设置一下他们和父节点的边`IndexedForwardGraph::Edge`，并没有往`graph_.post_dfs_oder`里面加节点。
首先，函数的参数没有父亲节点，并且是Opaque，函数的body也没有父亲节点。
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

这个是非常关键的一个函数，对于理解整个fusion非常重要。
首先是获取op到OpPattern的map。然后如果call的op是一个OpNode，获取其pattern.
然后update，设置其父亲为空，type为kOpaque。
之后遍历其所有的args。然后其args[i]的父亲都是这个call对应的node，并且对应的pattern设置为op_pattern。这就说明了Call是其args的父亲。
```C++
void VisitExpr_(const CallNode* call) final {
  ICHECK(graph_.node_map.count(call));
  Node* node = graph_.node_map.at(call);
  static auto fpattern = Op::GetAttrMap<TOpPattern>("TOpPattern");
  // Now we set the pattern of this call.
  //
  // If we see a call mentioning an operator we should mark it with its
  // annotated pattern.
  //
  // If the pattern is not annotated we will default to opaque.
  //
  // Finally if the operator position is not a call node we will
  // need to call Update, as it may be an arbitrary expression.
  OpPatternKind op_pattern = kOpaque;
  if (const OpNode* opnode = call->op.as<OpNode>()) {
    auto op = GetRef<Op>(opnode);
    if (IsDynamic(call->checked_type()) && IsDataDependent(call)) {
      // output of a shape func can't be fed to a data-dependent shape func
      op_pattern = kOpaque;
    } else {
      op_pattern = static_cast<OpPatternKind>(fpattern[op]);
    }
  } else {
    this->Update(call->op, node, kOpaque);
  }

  node->pattern = op_pattern;
  this->Update(call->op, nullptr, kOpaque);
  const auto* rtype = call->checked_type().as<TensorTypeNode>();
  // pass the analysis back to all the children it references.
  for (size_t i = 0; i < call->args.size(); ++i) {
    const auto* arg_type = call->args[i]->checked_type().as<TensorTypeNode>();
    // specifically check if result type is the same as arguments type
    OpPatternKind edge_pattern = op_pattern;
    if (edge_pattern == kBroadcast && arg_type != nullptr && rtype != nullptr &&
        attr_equal_(rtype->shape, arg_type->shape)) {
      edge_pattern = kElemWise;
    }
    this->Update(call->args[i], node, edge_pattern);
  }
  ExprVisitor::VisitExpr_(call);
  this->AddNode(call);
}
```

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


### 关于relay op中访问tir op
从relay op访问tir op中的属性，例如tir中的`ForNode`，访问其`min`，`extend`等属性。
首先看relay的`PRelue`是怎么定义的：
```C++
// Positional relay function to create prelu operator used by frontend FFI.
Expr MakePRelu(Expr data, Expr alpha, int axis) {
  auto attrs = make_object<PReluAttrs>();
  attrs->axis = axis;
  static const Op& op = Op::Get("nn.prelu");
  return Call(op, {data, alpha}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.prelu").set_body_typed(MakePRelu);

RELAY_REGISTER_OP("nn.prelu")
    .describe(R"code(Parametric version of a Rectified Linear Unit.
It accepts two arguments: an input ``x`` and a channelwise slope ``alpha``
and computes the output as :math:`PReLU(x) y = x > 0 ? x : alpha * x`,
where :math:`*` is an channelwise multiplication for each sample in the batch.
)code" TVM_ADD_FILELINE)
    .set_attrs_type<PReluAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "Input data.")
    .add_argument("alpha", "Tensor", "Input channelwise alpha.")
    .set_support_level(3)
    .add_type_rel("PRelu", PReluRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", PReluInferCorrectLayout)
    .set_attr<FTVMCompute>("FTVMCompute", [](const Attrs& attrs, const Array<te::Tensor>& inputs,
                                             const Type& out_type) {
      const auto* param = attrs.as<PReluAttrs>();
      return Array<te::Tensor>{topi::prelu(inputs[0], inputs[1], param->axis)};
    });
```
其中的`MakePRelu`返回了一个Relay的call，这里面的op就是下面`RELAY_REGISTER_OP`所定义的，tir的prelu,其一个属性为`FTVMCompute`就设置了其计算，调用的底层的`topi::prelu`，返回值是一个`te::Tensor`，其实现如下：
```C++
inline tvm::te::Tensor prelu(const tvm::te::Tensor& x, const tvm::te::Tensor& slope,
                             const int axis = 1, std::string name = "T_prelu",
                             std::string tag = kBroadcast) {
  ICHECK((size_t)axis < x->shape.size()) << "Wrong axis (" << axis << ")value. ";
  ICHECK(topi::detail::GetConstInt(slope->shape[0]) == topi::detail::GetConstInt(x->shape[axis]))
      << "Wrong slope shape received.";

  return tvm::te::compute(
      x->shape,
      [&](const tvm::Array<tvm::tir::Var>& indices) {
        auto xval = x(indices);
        return tvm::tir::Select(xval > 0, xval, xval * slope(indices[axis]));
      },
      name, tag);
}

/*!
 * \brief Computation description interface.
 *
 * \note This function have a special convention
 *  for functions with tuple input/output.
 *
 *  So far we restrict tuple support to the following case:
 *  - Function which takes a single tuple as input.
 *  - Function which outputs a single tuple.
 *
 *  In both cases, the tuple is flattened as array.
 *
 * \param attrs The attribute of the primitive
 * \param inputs The input tensors.
 * \param out_type The output type information
 &                 these are always placeholders.
 * \return The output compute description of the operator.
 */
using FTVMCompute = runtime::TypedPackedFunc<Array<te::Tensor>(
    const Attrs& attrs, const Array<te::Tensor>& inputs, const Type& out_type)>;
```
这个FTVMCompute就是tir的表示，因此能够调用到底层的tir.


### Relay op到底层的表示

```Python
 node[0], Var(data, ty=TensorType([1, 1, 4, 4], float32)) outputs=[2, 5, ]
 node[1], Constant([[[[1.]]]]) outputs=[2, ]
 node[2], CallNode(Op(nn.conv2d), [Var(data, ty=TensorType([1, 1, 4, 4], float32)), Constant([[[[1.]]]])], relay.attrs.Conv2DAttrs(0x55969e834208), [TensorType([1, 1, 4, 4], float32), TensorType([1, 1, 1, 1], float32)]) outputs=[3, ]
 node[3], CallNode(Op(nn.relu), [CallNode(Op(nn.conv2d), [Var(data, ty=TensorType([1, 1, 4, 4], float32)), Constant([[[[1.]]]])], relay.attrs.Conv2DAttrs(0x55969e834208), [TensorType([1, 1, 4, 4], float32), TensorType([1, 1, 1, 1], float32)])],   (nullptr), [TensorType([1, 1, 4, 4], float32)]) outputs=[7, ]
 node[4], Constant([[[[1.]]]]) outputs=[5, ]
 node[5], CallNode(Op(nn.conv2d), [Var(data, ty=TensorType([1, 1, 4, 4], float32)), Constant([[[[1.]]]])], relay.attrs.Conv2DAttrs(0x55969e79dde8), [TensorType([1, 1, 4, 4], float32), TensorType([1, 1, 1, 1], float32)]) outputs=[6, ]
 node[6], CallNode(Op(nn.relu), [CallNode(Op(nn.conv2d), [Var(data, ty=TensorType([1, 1, 4, 4], float32)), Constant([[[[1.]]]])], relay.attrs.Conv2DAttrs(0x55969e79dde8), [TensorType([1, 1, 4, 4], float32), TensorType([1, 1, 1, 1], float32)])],   (nullptr), [TensorType([1, 1, 4, 4], float32)]) outputs=[7, ]
 node[7], Tuple([CallNode(Op(nn.relu), [CallNode(Op(nn.conv2d), [Var(data, ty=TensorType([1, 1, 4, 4], float32)), Constant([[[[1.]]]])], relay.attrs.Conv2DAttrs(0x55969e834208), [TensorType([1, 1, 4, 4], float32), TensorType([1, 1, 1, 1],         float32)])], (nullptr), [TensorType([1, 1, 4, 4], float32)]), CallNode(Op(nn.relu), [CallNode(Op(nn.conv2d), [Var(data, ty=TensorType([1, 1, 4, 4], float32)), Constant([[[[1.]]]])], relay.attrs.Conv2DAttrs(0x55969e79dde8), [TensorType([1, 1, 4,  4], float32), TensorType([1, 1, 1, 1], float32)])], (nullptr), [TensorType([1, 1, 4, 4], float32)])]) outputs=[8, ]
 node[8], CallNode(Op(concatenate), 
    [Tuple([
      CallNode(Op(nn.relu), [
        CallNode(Op(nn.conv2d), [
          Var(data, ty=TensorType([1, 1, 4, 4], float32)), 
          Constant([[[[1.]]]])], 
          relay.attrs.Conv2DAttrs(0x55969e834208), [
            TensorType([1, 1, 4, 4], float32),       
            TensorType([1, 1, 1, 1], float32)])], (nullptr), [
              TensorType([1, 1, 4, 4], float32)]), 
      CallNode(Op(nn.relu), [
        CallNode(Op(nn.conv2d), [
          Var(data, ty=TensorType([1, 1, 4, 4], float32)), 
          Constant([[[[1.]]]])], relay.attrs.Conv2DAttrs(0x55969e79dde8), [TensorType([1, 1, 4, 4], float32), TensorType([1, 1, 1, 1], float32)])], (nullptr), [TensorType([1, 1, 4, 4], float32)])])], 
    relay.attrs.ConcatenateAttrs(0x55969e8eed88), 
    [TupleTypeNode([TensorType([1, 1, 4, 4],     float32), TensorType([1, 1, 4, 4], float32)])]) outputs=[]
```
目前的问题：
1. 如何判断op的类型并转成对应的relay op
2. 如何获取relay op中的各项属性
3. 如何获取到底层的tir的axis的表达式

可以通过如下来访问op的属性：
```C++
static auto fpattern = Op::GetAttrMap<TOpPattern>("TOpPattern");
op_pattern = static_cast<OpPatternKind>(fpattern[op]);
```
在`include/tvm/relay/op_attr_types.h`中定义了relay的op的各项属性，应该都可以通过如上的方式来获得。


### 关于Var和VarNode之间的互相转换（其他的IR也一样）
以`include/tvm/tir/var.h`中的`Var`为例：可以看到其`get()`方法返回值类型为`VarNode*`，其`->`操作返回值也是`VarNode*`。`const VarNode* get() const { return static_cast<const VarNode*>(data_.get()); }`
```C++
/*! \brief a named variable in TIR */
class Var : public PrimExpr {
 public:
  explicit Var(ObjectPtr<Object> n) : PrimExpr(n) {}
  /*!
   * \brief Constructor
   * \param name_hint variable name
   * \param dtype data type
   * \param span The location of this object in the source code.
   */
  TVM_DLL explicit Var(String name_hint = "v", DataType dtype = DataType::Int(32),
                       Span span = Span());
  /*!
   * \brief Constructor which provides a more detailed type annotation.
   * \param name_hint variable name.
   * \param type_annotation The type annotation.
   * \param span The location of this object in the source code.
   */
  TVM_DLL explicit Var(String name_hint, Type type_annotation, Span span = Span());

  /*!
   * \brief Get pointer to the internal value.
   * \return the corresponding Variable.
   */
  const VarNode* operator->() const { return get(); }
  /*!
   * \brief Get pointer to the internal value.
   * \return the corresponding Variable.
   */
  const VarNode* get() const { return static_cast<const VarNode*>(data_.get()); }
  /*! \brief type indicate the container type */
  using ContainerType = VarNode;
};
```


### relay pass 死代码消除的代码解读
入口代码如下：
```C++
static Expr Eliminate(const Expr& e, bool inline_once) {
    FindDef fd;
    fd(e);
    CalcDep cd(fd.expr_map_);
    cd(e);
    Eliminator el(fd.expr_map_, cd.use_map_, inline_once);
    return el(e);
  }
```
首先定义了一个VarMap
`using VarMap = std::unordered_map<Var, X, ObjectPtrHash, ObjectPtrEqual>;`
VarMap存的是LetNode中的var到value的映射。在pre_visit中加入映射，并且递归的访问op->value；在post_visit中访问op->body，并且记visit_counter_[op]+=1。
ExpandANormalForm，简单来说就是如果op->body还是LetNode，那就不断push，直到不是LetNode了，就不断弹出，并且在post_visit里面记录访问的次数。
```C++
void ExpandANormalForm(const LetNode* op, std::function<void(const LetNode*)> pre_visit,
                       std::function<void(const LetNode*)> post_visit) {
  std::stack<const LetNode*> stack;
  stack.push(op);
  bool is_anormal = true;
  while (is_anormal) {
    const LetNode* current_op = stack.top();
    pre_visit(current_op);
    if (const LetNode* new_op = current_op->body.as<LetNode>()) {
      stack.push(new_op);
    } else {
      is_anormal = false;
    }
  }
  while (stack.size()) {
    const LetNode* current_op = stack.top();
    stack.pop();
    post_visit(current_op);
  }
}
```

在CalcDep里面，如果访问LetNode,不断的获取let的body,如果是一个let_binding，就增加外层入口的letNode的visit_counter，然后递归执行。
如果是一个VarNode，说明这个var被使用，那么LetNode中原来这个var对应的expr就需要被访问。
如果是到了Leaf叶子结点，看visit_counter_，如果<=2，又调用了一个啥，VisitLead是在哪里被调用的待搞明白。
最后是Eliminator,首先看起VisitExpr_的VarNode，如果exprt_map_中没有v，则返回v，否则继续访问其对应的Expr。
如果是LetNode（1）prev_isit中如果op->var hasLet，继续访问op->value；在post_visit中，首先visit他的body,之后获取op->var，如果这个v hasLet，则visit(op->value)，兵器重新创建一个Let。否则，只记录这个`this->memo_[expr] = body;`，ExpandANormalForm仍然是上面的操作。最后返回的是memo_中的op对应的。


### Relay获取op的attr，以及插入op的操作。
在`src/relay/transforms/auto_scheduler_layout_rewrite.cc`中对于Conv会插入更改其layout的算子。只看核心部分的操作：一样的套路如果是CallNode，并且call->op为OpNode，则创建一个新的CallNode，这个CallNode的op为`Op::Get("auto_scheduler_layout_transform")`，args为data（其实就是Conv的weight），attr就是新的layout。然后更新CallOp原来的args。通过这个attr看看能不能获得tensor的shape大小。

```C++

Expr MakeAutoSchedulerLayoutTransform(Expr data, String src_layout, String dst_layout) {
  auto attrs = make_object<AutoSchedulerLayoutTransformAttrs>();
  attrs->src_layout = std::move(src_layout);
  attrs->dst_layout = std::move(dst_layout);
  static const Op& op = Op::Get("auto_scheduler_layout_transform");
  return Call(op, {data}, Attrs(attrs), {});
}

Expr VisitExpr_(const CallNode* n) {
  auto new_n = ExprMutator::VisitExpr_(n);

  const auto* call = new_n.as<CallNode>();
  if (call && call->op.as<OpNode>() &&
      (std::find(target_ops_.begin(), target_ops_.end(), n->op.as<OpNode>()->name) !=
        target_ops_.end()) &&
      !ori_layouts_queue_.empty() && !new_layouts_queue_.empty()) {
    // Pop a new layout from the queue
    const std::string ori_layout = ori_layouts_queue_.front();
    const std::string new_layout = new_layouts_queue_.front();
    ori_layouts_queue_.pop_front();
    new_layouts_queue_.pop_front();

    // Insert a new op to do layout transform. (This will be simplified by FoldConstant later).
    Expr updated_kernel = MakeAutoSchedulerLayoutTransform(call->args[1], ori_layout, new_layout);
    Array<Expr> updated_args = {call->args[0], updated_kernel};

    // Update the attrs
    Attrs updated_attrs;
    if (auto pattr = call->attrs.as<Conv2DAttrs>()) {
      updated_attrs = CopyAttrsWithNewLayout(pattr, new_layout);
    } else if (auto pattr = call->attrs.as<Conv2DWinogradAttrs>()) {
      updated_attrs = CopyAttrsWithNewLayout(pattr, new_layout);
    } else if (auto pattr = call->attrs.as<Conv3DAttrs>()) {
      updated_attrs = CopyAttrsWithNewLayout(pattr, new_layout);
    } else if (auto pattr = call->attrs.as<MatmulAttrs>()) {
      updated_attrs = CopyAttrsWithNewLayout(pattr, new_layout);
    } else if (auto pattr = call->attrs.as<DenseAttrs>()) {
      updated_attrs = CopyAttrsWithNewLayout(pattr, new_layout);
    } else if (auto pattr = call->attrs.as<BatchMatmulAttrs>()) {
      updated_attrs = CopyAttrsWithNewLayout(pattr, new_layout);
    } else {
      LOG(FATAL) << "Unhandled attribute: " << call->attrs;
    }
    new_n = Call(call->op, updated_args, updated_attrs);
  }
  return new_n;
}
```

### TVM中合并并行的conv和matmul
在
以Convd为例,Finder当中的Combine函数首先用`BranchGroupFinder`来找可以合并的conv，
之后分别把每个group中的算子合并。`CanOpsBeCombined`就是检查CallNode的args是否一样多，每个args的number of dimension是否一样。
```C++
Expr CombineParallelConv2D(const Expr& expr, uint64_t min_num_branches) {
  return ParallelConv2DCombiner(min_num_branches).Combine(expr);
}

explicit ParallelConv2DCombiner(uint64_t min_num_branches)
    : ParallelOpCombiner("nn.conv2d", min_num_branches) {}

ParallelOpCombiner::ParallelOpCombiner(const std::string& op_name, uint64_t min_num_branches)
    : cached_op_(Op::Get(op_name)), min_num_branches_(min_num_branches) {}

Expr ParallelOpCombiner::Combine(const Expr& expr) {
  auto groups = BranchGroupFinder(
                    cached_op_, [&](const CallNode* n) { return IsSupportedOp(n); },
                    [&](const CallNode* a, const CallNode* b) { return CanOpsBeCombined(a, b); })
                    .Find(expr);
  for (const Group& group : groups) {
    if (group.size() < min_num_branches_) {
      continue;
    }
    CombineBranches(group);
  }
  return ExprSubst(expr, std::move(subst_map_));
}
```

find函数如下：首先是调了VisitExpr，针对所有的CallNode，如果是要fuse的op，记到op_roots里面，这句话：如果op是conv并且受支持，则op_root就插入其n->args[0]（就是conv的input）
```C++
if (n->op == cached_op_ && fis_supported_op_(n)) {
  op_roots_.insert(n->args[0]);
  children_map_[n->args[0]].push_back(n);
}
```
然后是其var的孩子`children`，如果不是conv就跳过。之后是创造conv之后，conv后面的一串op凡是其属性为`kBroadcast`一下的，就往后继续添加到一个branch里面。在所有的group中找能够与当前branch fuse的（见`CanOpsBeCombined`函数，以conv2d为例就是比较两个node的layout，除了output channels之外所有的参数都要相同）。把这个branch放到一个group中。
```C++
std::vector<Group> BranchGroupFinder::Find(const Expr& expr) {
  this->VisitExpr(expr);

  std::vector<Group> groups;
  for (const auto& root : op_roots_) {
    const auto& children = children_map_.at(root);
    size_t ngroups = groups.size();
    for (const CallNode* child : children) {
      if (child->op != cached_op_) continue;

      auto&& branch = CreateBranch(child);
      // add the branch to a group, or create a new group
      auto it = std::find_if(groups.begin() + ngroups, groups.end(), [&](const Group& group) {
        ICHECK(!group.empty() && !group[0].empty());
        return fare_compatible_ops_(child, group[0][0]);
      });
      if (it != groups.end()) {
        it->push_back(branch);
      } else {
        groups.emplace_back();
        // each group has at least one branch
        groups.back().push_back(branch);
      }
    }
  }
  return groups;
}
```

之后是合并每个group中的branch。仍然以Conv2D为例，`TransformWeight`中把`branch`中conv的weight构成一个concate，然后就是建立了一个Conv2d的op，设置参数。
```C++
Call MakeCombinedOp(const Group& branches) {
  const Op& conv2d = Op::Get("nn.conv2d");
  Expr data = branches[0][0]->args[0];
  Expr new_weight;
  IndexExpr new_channels;
  std::tie(new_weight, new_channels) = TransformWeight(branches);

  const CallNode* group_root = branches[0][0];
  const auto* attrs = group_root->attrs.as<Conv2DAttrs>();
  ICHECK(attrs);
  const auto new_attrs = make_object<Conv2DAttrs>();
  new_attrs->strides = attrs->strides;
  new_attrs->padding = attrs->padding;
  new_attrs->dilation = attrs->dilation;
  new_attrs->groups = attrs->groups;
  new_attrs->kernel_size = attrs->kernel_size;
  new_attrs->data_layout = attrs->data_layout;
  new_attrs->kernel_layout = attrs->kernel_layout;
  new_attrs->out_layout = attrs->out_layout;
  new_attrs->out_dtype = attrs->out_dtype;
  new_attrs->channels = new_channels;

  const std::string& layout =
      new_attrs->out_layout == "" ? new_attrs->data_layout : new_attrs->out_layout;
  channel_pos_ = layout.find('C');
  ICHECK_NE(channel_pos_, std::string::npos);

  return Call(conv2d, {data, new_weight}, Attrs{new_attrs}, {});
}
```
`CombineBranches`是horizontal fusion的主函数。`MakeCombinedOp`把每个branch第一个op（也就是conv）给合并到一起。然后看看各个branch最低的深度，然后`CheckLevel`所有branch的每个层逐一检查，所有的op必须是相同的op，并且属性必须相同，并且只能有一个子节点，并且其输入参数要类型一致，这样才能`MakeCombinedCallFromFollowingOps`, 这个函数的做法是把同一level的op的所有args的所有branch的arg都Stack到一起（注意op可能会有多个args，因此最后合并生成的op的arg）。最后`UpdateGroupOutput`是根据conv2d的output chennel的大小加入slice
```C++
void ParallelOpCombiner::CombineBranches(const Group& branches) {
  Call combined = MakeCombinedOp(branches);
  auto it = std::min_element(branches.begin(), branches.end(),
                             [](const Branch& branch_a, const Branch& branch_b) {
                               return branch_a.size() < branch_b.size();
                             });
  size_t depth = it->size();
  size_t i;
  // starting from 1 to skip the op
  for (i = 1; i < depth; i++) {
    size_t parent_index;
    for (parent_index = 0; parent_index < branches[0][i]->args.size(); parent_index++) {
      if (branches[0][i]->args[parent_index].get() == branches[0][i - 1]) break;
    }
    ICHECK_NE(parent_index, branches[0][i]->args.size());
    if (!CheckLevel(branches, i, parent_index)) break;
    combined = MakeCombinedCallFromFollowingOps(combined, branches, i, parent_index);
  }
  UpdateGroupOutput(combined, branches, i - 1, &subst_map_);
}
```

### 关于打印relay node的函数

在`src/relay/ir/expr.cc`中以LetNode为例：
```C++
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<LetNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const LetNode*>(ref.get());
      p->stream << "LetNode(" << node->var << ", " << node->value << ", " << node->body << ")";
    });
```

### 从relay的IR变换到TIR的过程
在`python/tvm/ir/module.py`中，
```C++
    @staticmethod
    def from_expr(expr, functions=None, type_defs=None):
        """Construct a module from a standalone expression.

        Parameters
        ----------
        expr: RelayExpr
            The starting expression

        global_funcs: Optional[dict]
            Map of global vars to function definitions

        type_defs: Optional[dict]
            Map of global type vars to type definitions

        Returns
        -------
        mod: Module
            A module containing the passed definitions,
            where expr is set as the entry point
            (wrapped in a function if necessary)
        """
        funcs = functions if functions is not None else {}
        defs = type_defs if type_defs is not None else {}
        return _ffi_api.Module_FromExpr(expr, funcs, defs)
```

之后在`src/ir/module.cc`中，`TVM_REGISTER_GLOBAL("ir.Module_FromExpr").set_body_typed(&IRModule::FromExpr);`
```C++

IRModule IRModule::FromExpr(const RelayExpr& expr, const Map<GlobalVar, BaseFunc>& global_funcs,
                            const Map<GlobalTypeVar, TypeData>& type_definitions) {
  return FromExprInContext(expr, global_funcs, type_definitions).first;
}
```

```C++
std::pair<IRModule, GlobalVar> IRModule::FromExprInContext(
    const RelayExpr& expr, const tvm::Map<GlobalVar, BaseFunc>& global_funcs,
    const tvm::Map<GlobalTypeVar, TypeData>& type_definitions,
    std::unordered_set<String> import_set) {
  auto mod = IRModule(global_funcs, type_definitions, std::move(import_set));
  String gv_name;

  // All global definitions must be functions.
  BaseFunc func;
  if (auto* func_node = expr.as<BaseFuncNode>()) {
    func = GetRef<BaseFunc>(func_node);
    if (auto opt = func->GetAttr<String>(tvm::attr::kGlobalSymbol)) {
      // Function literal has been annotated with it's required global symbol.
      gv_name = opt.value();
    }
  } else {
    func = relay::Function(relay::FreeVars(expr), expr, Type(), relay::FreeTypeVars(expr, mod), {});
  }

  if (gv_name.empty()) {
    // Bind function to 'main' (though rename if would clash with existing 'main').
    gv_name = mod->GetUniqueName("main");
  }

  GlobalVar main_gv(gv_name);
  mod->Add(main_gv, func);
  return {mod, main_gv};
}
```

未完待续

### 如何从relay的op到TIR的
首先从relay的add,其属性FTVMCompute为一个PackFunction, 类型为PackedFunc，描述了计算是什么，
这个function会读入一个tuple的tensor作为input，返回一个tulple的tensor作为output。

```C++
// Addition
RELAY_REGISTER_BINARY_OP("add")
    .describe("Elementwise add with broadcasting")
    .set_support_level(1)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::add));
```

```C++
using FTVMCompute = runtime::TypedPackedFunc<Array<te::Tensor>(
    const Attrs& attrs, const Array<te::Tensor>& inputs, const Type& out_type)>;
```
```C++
Add::Add(PrimExpr a, PrimExpr b, Span span) { 
  using T = Add::ContainerType; 
  ICHECK(a.defined()) << "ValueError: a is undefined\n"; 
  ICHECK(b.defined()) << "ValueError: b is undefined\n"; 
  ICHECK(a.dtype() == b.dtype()) << "TypeError: mismatched types. " << a.dtype() << " vs. " << b.dtype() << "\n"; 
  ObjectPtr<T> node = make_object<T>(); 
  node->dtype = a.dtype(); node->a = std::move(a); 
  node->b = std::move(b); node->span = std::move(span); 
  data_ = std::move(node); 
  }
```

底层的IRModuleNode:
```C++
void IRModuleNode::Add(const GlobalVar& var, const BaseFunc& f, bool update) {
  BaseFunc checked_func = f;
  if (auto* ptr = f.as<relay::FunctionNode>()) {
    WarnIfMalformed(GetRef<IRModule>(this), GetRef<relay::Function>(ptr));
  }

  AddUnchecked(var, checked_func);
}
```

在`src/tir/ir/expr.cc`中有Add的定义。
```C++
TVM_DEFINE_BINOP_CONSTRUCTOR(Add);

TVM_REGISTER_GLOBAL("tir.Add").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return Add(a, b, span);
});

TVM_REGISTER_NODE_TYPE(AddNode);

```
展开
```C++
Add::Add(PrimExpr a, PrimExpr b, Span span) { 
  using T = Add::ContainerType; 
  ICHECK(a.defined()) << "ValueError: a is undefined\n"; ICHECK(b.defined()) << "ValueError: b is undefined\n"; 
  ICHECK(a.dtype() == b.dtype()) << "TypeError: mismatched types. " << a.dtype() << " vs. " << b.dtype() << "\n"; 
  ObjectPtr<T> node = make_object<T>(); 
  node->dtype = a.dtype(); node->a = std::move(a); node->b = std::move(b); 
  node->span = std::move(span);
  data_ = std::move(node); 
}

TVM_STR_CONCAT(TVM_FUNC_REG_VAR_DEF, __COUNTER__) = ::tvm::runtime::Registry::Register("tir.Add")


TVM_REGISTER_OBJECT_TYPE(AddNode); 
TVM_REGISTER_REFLECTION_VTABLE(AddNode, ::tvm::detail::ReflectionTrait<AddNode>) .set_creator([](const std::string&) -> ObjectPtr<Object> { return ::tvm::runtime::make_object<AddNode>(); })
```

终于找到在哪里把relay的function给lower成TIR了！
在`src/relay/backend/te_compiler.h`中！
有理由怀疑`src/relay/backend/build_module.cc`就是入口。下面这个函数就是直接从Python中调过来的。之后是在`BuildRelay`的函数中。
```C++
  /*!
   * \brief Build relay IRModule for graph executor
   *
   * \param mod Relay IRModule
   * \param target Target device
   * \param target_host Host target device
   */
  void Build(IRModule mod, const TargetsMap& targets, const tvm::Target& target_host,
             const String executor, const String mod_name) {
    for (const auto& pair : targets) {
      VLOG(0) << "Build target " << pair.first << " = " << pair.second->str();
    }
    if (target_host.defined()) {
      VLOG(0) << "Build target_host = " << target_host->str();
    }
    VLOG(0) << "Build executor = '" << executor << "'";
    VLOG(0) << "Build mod_name = '" << mod_name << "'";

    // Create protected variable targets_ from ground up
    targets_ = targets;
    target_host_ = target_host;
    executor_ = executor;
    CheckAndUpdateHostConsistency(&targets_, &target_host_);
    BuildRelay(mod, params_, mod_name);
  }
```
省略不重要的，在`MakeExecutorCodegen`中确认目前是调用的`GraphCodegen`。
Init函数都是一样继承自`ExecutorCodegen`，只看`Codegen`函数。
```C++
/*!
  * \brief Compile a Relay IR module to runtime module.
  *
  * \param relay_module The Relay IR module.
  * \param params The parameters.
  */
void BuildRelay(IRModule relay_module,
                const std::unordered_map<std::string, tvm::runtime::NDArray>& params,
                const String mod_name) {
  Target target_host = GetTargetHost();
  // If no target_host has been set, we choose a default one, which is
  // llvm if "codegen.LLVMModuleCreate" is accessible.
  const runtime::PackedFunc* pf = runtime::Registry::Get("codegen.LLVMModuleCreate");
  if (!target_host.defined()) target_host = (pf != nullptr) ? Target("llvm") : Target("stackvm");

  // Update all the targets in the targets_ TargetsMap
  CheckAndUpdateHostConsistency(&targets_, &target_host);

  // Relay IRModule -> IRModule optimizations.
  relay_module = OptimizeImpl(relay_module, params);

  // Get the updated function.
  auto func = Downcast<Function>(relay_module->Lookup("main"));

  // Generate code for the updated function.
  executor_codegen_ = MakeExecutorCodegen(executor_);
  executor_codegen_->Init(nullptr, targets_);
  executor_codegen_->Codegen(func, mod_name);
  executor_codegen_->UpdateOutput(&ret_);
  ret_.params = executor_codegen_->GetParams();

  auto lowered_funcs = executor_codegen_->GetIRModule();

  ...
}
```

`Codegen`函数在`src/relay/backend/graph_executor_codegen.cc`中，
```C++
TVM_REGISTER_GLOBAL("relay.build_module._GraphExecutorCodegen")
    .set_body([](TVMArgs args, TVMRetValue* rv) { *rv = CreateGraphCodegenMod(); });
```

```C++
runtime::Module CreateGraphCodegenMod() {
  auto ptr = make_object<GraphExecutorCodegenModule>();
  return runtime::Module(ptr);
}
```
在GraphExecutorCodegen中：
```C++
/*! \brief Code generator for the graph executor, produces a module containing the graph JSON,
 * module, and parameters.
 */
class GraphExecutorCodegen : public backend::MemoizedExprTranslator<std::vector<GraphNodeRef>> {
 public:
  GraphExecutorCodegen(runtime::Module* mod, const tec::TargetMap& targets) : mod_(mod) {
    targets_ = targets;
  }

  StorageInfo GetStorageInfo(const Expr& e) {
    size_t count = memory_plan_->expr_to_storage_info.count(e);
    ICHECK_GT(count, 0) << "Expr is not existing in storage plan";
    auto storage_info = memory_plan_->expr_to_storage_info[e];
    return storage_info;
  }

  LoweredOutput Codegen(relay::Function func, String mod_name) {
    mod_name_ = mod_name;
    VLOG_CONTEXT << "GraphExecutorCodegen";
    VLOG(1) << "compiling:" << std::endl << PrettyPrint(func);
    for (const auto& pair : targets_) {
      VLOG(1) << "target: " << pair.first << " = " << pair.second->str();
    }

    // This first phase moves from implicit use of compile engine,
    // to instead explicitly lowering the incoming IRModule, and then
    // performing the preexisting graph executor code generation phase.
    IRModule mod = IRModule::FromExpr(func);

    // TODO(mbs): Why plan memory and update workspace sizes before lowering?
    memory_plan_ = GraphPlanMemory(func);

    backend::FunctionInfo func_info;

    if (memory_plan_.defined()) {
      // TODO(@electriclilies, @jroesch): remove UpdateMainWorkspaceSize
      func_info =
          relay::tec::UpdateMainWorkspaceSize(mod, targets_, memory_plan_->expr_to_storage_info);
      mod = WithAttr(mod, "main_func_info", func_info);
    }

    IRModule lowered_mod = tec::LowerTEPass(targets_, mod_name_, [this](Function func) {
      // We need to maintain the constant map for external
      // functions so we pass this processing function which
      // allows us to process each function as we lower it.
      if (func->GetAttr<String>(attr::kCompiler).defined()) {
        UpdateConstants(func, &params_);
      }

      // TODO(@areusch, @jroesch): We should refactor this to
      // execute as a further pass, instead writing data to the
      // lowering process directly.
      tec::UpdateFunctionMetadata(func, this->function_metadata_);
    })(mod);

    Optional<backend::FunctionInfo> main_func_info =
        lowered_mod->GetAttr<backend::FunctionInfo>("main_func_info");

    function_metadata_.Set(runtime::symbol::tvm_module_main, main_func_info.value());

    Function lowered_main_func = Downcast<Function>(lowered_mod->Lookup("main"));

    // Now that we have lowered all operators to TIR code, we can proceed with compilation.
    //
    // We need to unfortunately re-plan as the previous results have been invalidated by lowering
    // we will fix this in future refactors.
    memory_plan_ = GraphPlanMemory(lowered_main_func);

    // The graph planner also can not handle planning calls to global variables to we must remap

    // First we convert all the parameters into input nodes.
    for (auto param : lowered_main_func->params) {
      auto node_ptr = GraphInputNode::make_node_ptr(param->name_hint(), GraphAttrs());
      var_map_[param.get()] = AddNode(node_ptr, param);
    }

    heads_ = VisitExpr(lowered_main_func->body);
    std::ostringstream os;

    dmlc::JSONWriter writer(&os);
    GetJSON(&writer);
    LoweredOutput ret;
    ret.graph_json = os.str();
    ret.params = std::unordered_map<std::string, std::pair<int, const tvm::runtime::NDArray>>();
    for (auto param : params_) {
      ret.params.emplace(std::make_pair(
          param.first,
          std::make_pair(static_cast<int>(param_storage_ids_[param.first]), param.second)));
    }
    ret.function_metadata = std::move(function_metadata_);

    Optional<Array<tvm::runtime::Module>> external_modules =
        lowered_mod->GetAttr<Array<tvm::runtime::Module>>("external_mods");
    ICHECK(external_modules) << "Attribute \"external_mods\" should be set at this point.";

    // This is the point where we separate the functions in the module by target
    ret.lowered_funcs = tec::GetPerTargetModules(lowered_mod);
    ret.external_mods = external_modules.value();
    return ret;
  }
  ...
}
```


再转到`src/relay/backend/te_compiler.cc`中的`Pass LowerTEPass`中。
```C++
Pass LowerTEPass(TargetMap targets, const String& module_name,
                 std::function<void(Function)> process_fn) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule module,
                                                                            PassContext ctx) {
    return LowerTE(module, targets, module_name, process_fn);
  };

  return tvm::transform::Sequential({tvm::relay::transform::RelayToTIRTargetHook(),
                                     tvm::transform::CreateModulePass(pass_func, 0, "LowerTE", {}),
                                     InferType()});
}
```

```C++
IRModule LowerTE(const IRModule& module, TargetMap targets, const String& module_name,
                 std::function<void(Function)> process_fn) {
  DLOG(INFO) << "lowering module:\n" << PrettyPrint(module);

  TECompiler compiler;

  auto updated_module = LowerTensorExpr(targets, module_name, compiler, process_fn)(module);

  backend::UpdateAutoSchedulerOpWeights(compiler);

  // Copy the lowered functions into the return module
  updated_module->Update(compiler->GetLoweredFunctions());

  // Annotate the module with the external modules and function info
  updated_module = WithAttr(updated_module, "external_mods", compiler->LowerExternalFunctions());

  return updated_module;
}
```

之后是FunctionPassNode的operator(),144行
```C++
// Perform Module -> Module optimizations at the Function level.
IRModule FunctionPassNode::operator()(IRModule mod, const PassContext& pass_ctx) const {
  DiagnosticContext previous = DiagnosticContext::Default(mod);

  if (pass_ctx->diag_ctx) {
    DiagnosticContext tmp = pass_ctx->diag_ctx.value();
    pass_ctx->diag_ctx = previous;
    previous = tmp;
  } else {
    pass_ctx->diag_ctx = previous;
  }

  ICHECK(pass_ctx->diag_ctx)
      << "The diagnostic context was set at the top of this block this is a bug.";

  const PassInfo& pass_info = Info();

  ICHECK(mod.defined());

  VLOG_CONTEXT << pass_info->name;
  VLOG(0) << "Executing function pass with opt level: " << pass_info->opt_level;
  VLOG(1) << "Input module:" << std::endl << PrettyPrint(mod);

  IRModule updated_mod = mod->ShallowCopy();

  std::vector<std::pair<GlobalVar, Function> > updates;
  for (const auto& it : updated_mod->functions) {
    // only picks up relay::Function
    if (auto* n = it.second.as<FunctionNode>()) {
      Function func = GetRef<Function>(n);
      auto updated_func = SkipFunction(func) ? func : pass_func(func, updated_mod, pass_ctx);
      updates.push_back({it.first, updated_func});
    }
  }

  for (const auto& pair : updates) {
    updated_mod->Add(pair.first, pair.second, true);
  }

  ICHECK(pass_ctx->diag_ctx)
      << "The diagnostic context was set at the top of this block this is a bug.";

  pass_ctx->diag_ctx.value().Render();
  pass_ctx->diag_ctx = previous;

  VLOG(1) << "Output module:" << std::endl << PrettyPrint(updated_mod);

  // TODO(@jroesch): move away from eager type checking for performance reasons
  // make issue.
  return transform::InferType()(updated_mod);
}

```

之后是
```C++
Pass LowerTensorExpr(TargetMap targets, const String& module_name, TECompiler compiler,
                     std::function<void(Function)> process_fn) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function func, IRModule module, PassContext ctx) {
        LowerTensorExprMutator lower_te(module, targets, process_fn, module_name, compiler);
        return Downcast<Function>(lower_te.Mutate(func));
      };
  return CreateFunctionPass(pass_func, 0, "LowerTensorExpr", {});
}
```


`src/relay/transforms/device_aware_visitors.cc:217`,在PushBoudVar中，是如何知道function_node->params[i]是var类型的？
```C++
Expr DeviceAwareExprMutator::VisitExpr_(const FunctionNode* function_node) {
  if (function_node->HasNonzeroAttr(attr::kPrimitive)) {
    // No tracking inside primitive functions.
    return DeviceAwareVisitExpr_(function_node);
  } else {
    // Function parameters come into scope.
    for (size_t i = 0; i < function_node->params.size(); ++i) {
      PushBoundVar(function_node->params[i], GetFunctionParamDeviceType(function_node, i));
    }
    // Entering scope of function body.
    PushDeviceType(GetFunctionResultDeviceType(function_node));
    EnterFunctionBody();

    Expr result = DeviceAwareVisitExpr_(function_node);

    // Leaving scope of function body.
    ExitFunctionBody();
    PopDeviceType();
    // Function parameters go out of scope.
    for (size_t i = 0; i < function_node->params.size(); ++i) {
      PopBoundVar(function_node->params[i]);
    }

    return result;
  }
}

```

这里面分别Mutate(param)和Mutate(op->body)。如果没有变化，则还返回原来的，否则，返回新的Function.
```C++
Expr ExprMutator::VisitExpr_(const FunctionNode* op) {
  tvm::Array<TypeVar> ty_params;
  bool all_ty_params_unchanged = true;

  for (auto ty_param : op->type_params) {
    TypeVar new_ty_param = Downcast<TypeVar>(VisitType(ty_param));
    ty_params.push_back(new_ty_param);
    all_ty_params_unchanged &= new_ty_param.same_as(ty_param);
  }

  tvm::Array<Var> params;
  bool all_params_unchanged = true;
  for (auto param : op->params) {
    Var new_param = Downcast<Var>(this->Mutate(param));
    params.push_back(new_param);
    all_params_unchanged &= param.same_as(new_param);
  }

  auto ret_type = this->VisitType(op->ret_type);
  auto body = this->Mutate(op->body);

  if (all_ty_params_unchanged && all_params_unchanged && ret_type.same_as(op->ret_type) &&
      body.same_as(op->body)) {
    return GetRef<Expr>(op);
  } else {
    return Function(params, body, ret_type, ty_params, op->attrs, op->span);
  }
}
```

```C++
Expr DeviceAwareExprMutator::VisitExpr_(const CallNode* call_node) {
  auto props = GetOnDeviceProps(call_node);
  if (props.body.defined() && props.is_fixed) {
    // Entering lexical scope of fixed "on_device" call.
    PushDeviceType(props.device_type);
    Expr expr = VisitExpr(props.body);
    // Leaving lexical scope of "on_device" call.
    PopDeviceType();
    return OnDevice(expr, props.device_type, props.is_fixed);
  } else {
    return DeviceAwareVisitExpr_(call_node);
  }
}
```
然后调到`LowerTensorExprMutator`。
这个类继承了`DeviceAwareExprMutator`, 其祖父类为`ExprMutator`、`LexicalOnDeviceMixin`
下面这个是一个核心函授，首先看call_node的op是否能被解析为PrimiFunc，这是只包含TIR的。这里面不断的lower PrimFunc
```C++
Expr DeviceAwareVisitExpr_(const CallNode* call_node) override {
  Call call = GetRef<Call>(call_node);
  // Look for (indirect) calls to primitives.
  BaseFunc prim_func = ResolveToPrimitive(call_node->op);
  if (!prim_func.defined()) {
    // Not a call_node to a primitive function.
    if (const FunctionNode* fn = call_node->op.as<FunctionNode>()) {
      this->process_fn_(GetRef<Function>(fn));
    }
    return ExprMutator::VisitExpr_(call_node);
  }

  // Already lowered by other means so we don't need to mutate
  // the call
  if (prim_func->IsInstance<tir::PrimFuncNode>()) {
    return std::move(call);
  }

  // Find the desired target device.
  Target target;
  if (prim_func->GetAttr<String>(attr::kCompiler).defined()) {
    // The generic 'external device' target.
    target = Target("ext_dev");
  } else {
    // The target corresponding to the call_node expression's annotation.
    DLDeviceType device_type = GetInScopeDeviceType(call);
    // TODO(mbs): Replace device_type with target so this lookup is unnecessary.
    target = GetTargetFromInteger(device_type, targets_);
  }

  // Lower the primitive function for that target.
  Function func = Downcast<Function>(prim_func);
  std::pair<GlobalVar, Attrs> pair = LowerFunction(func, target);

  // Similarly transform arguments.
  Array<Expr> args;
  for (const auto& arg : call_node->args) {
    args.push_back(VisitExpr(arg));
  }

  // Replace with direct call to lowered primitive, and attach annotations to record calling
  // convention.
  return Call(pair.first, args, pair.second);
}
```

在`src/relay/backend/te_compiler.cc`中，LowerInternal中调了lower_schedule，应该是一个主要的函数。传进来的CCacheKey中包含的是relay的function和target，因此如果有相同的已经被生成，就不需要再次调用lower的过程量，直接返回即可。之后的`PrimFuncFor`就是最核心的函数。其从Function中产生了ComputeOp，根据ComputeOp的TE已经对应的strategy产生schedule。以及调用LowerSchedule生成底层的function。
```C++
CCacheValue LowerInternal(const CCacheKey& key, std::function<String(String)> mangle_fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    CCacheValue value;
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      it->second->use_count += 1;
      if (it->second->cached_func.defined()) return it->second;
      value = it->second;
    } else {
      value = CCacheValue(make_object<CCacheValueNode>());
      value->use_count = 1;
      cache_[key] = value;
    }
    cur_ccache_key_ = key;

    // No need to lower external functions for now. We will invoke the external
    // codegen tool once and lower all functions together.
    if (key->source_func->GetAttr<String>(attr::kCompiler).defined()) {
      auto ir_module = IRModule();
      const auto name_node = key->source_func->GetAttr<String>(tvm::attr::kGlobalSymbol);
      ICHECK(name_node.defined()) << "External function has not been attached a name yet.";
      auto func_name = GetUniqueName(name_node.value(), &name_map_);
      auto target = Target("ext_dev");
      auto global_var = GlobalVar(func_name);
      global_var->checked_type_ = key->source_func->checked_type();
      ir_module->Add(global_var, key->source_func);
      value->cached_func = CachedFunc(target, global_var, {}, {}, te::Schedule(), {}, ir_module);
      return value;
    }

    // Enforce use the target.
    With<Target> target_scope(key->target);

    ICHECK(!value->cached_func.defined());
    auto cfunc = PrimFuncFor(key->source_func, key->target, [&](std::string name) {
      auto mangled = mangle_fn(name);
      return GetUniqueName(mangled, &name_map_);
    });

    // Skip lowering for device copy node.
    const Expr body = (key->source_func)->body;
    if (const CallNode* call_node = body.as<CallNode>()) {
      if (call_node->attrs.as<DeviceCopyAttrs>()) {
        value->cached_func = cfunc;
        return value;
      }
    }

    // NOTE: array will copy on write.
    Array<te::Tensor> all_args = Array<te::Tensor>(cfunc->inputs);
    for (te::Tensor arg : cfunc->outputs) {
      all_args.push_back(arg);
    }

    std::unordered_map<te::Tensor, tir::Buffer> binds;
    auto func_name = cfunc->prim_fn_var->name_hint;
    cfunc->funcs->Update(tvm::LowerSchedule(cfunc->schedule, all_args, func_name, binds));
    value->cached_func = cfunc;
    return value;
  }
```

后面的PrimFuncFor是为这个函数创建一个默认的schedule。
```C++
/*!
 * \brief Create schedule for target.
 * \param source_func The primitive function to be lowered.
 * \param target The target we want to create schedule for.
 * \return Pair of schedule and cache.
 *  The funcs field in cache is not yet populated.
 */
CachedFunc PrimFuncFor(const Function& source_func, const Target& target,
                       std::function<std::string(std::string)> renamer) {
  return ScheduleBuilder(target).Create(source_func, renamer);
}
```
```C++
CachedFunc Create(const Function& prim_func, std::function<std::string(std::string)> renamer) {
  Array<tvm::te::Tensor> fn_inputs;
  for (Var param : prim_func->params) {
    Array<tvm::te::Tensor> inputs;
    for (const auto& ttype : FlattenTupleType(param->checked_type())) {
      tvm::te::Tensor tensor = tvm::te::placeholder(GetShape(ttype->shape), ttype->dtype);
      fn_inputs.push_back(tensor);
      inputs.push_back(tensor);
    }
    memo_[param] = inputs;
  }
  readable_name_stream_ << "fused";
  auto outputs = this->VisitExpr(prim_func->body);
  auto candidate_name = readable_name_stream_.str();
  constexpr static size_t kMaxFuncNameLength = 80;
  // WARNING: Please make sure to also update TVM_CRT_MAX_STRLEN_FUNCTION_NAME
  //          whenever the value of kMaxFuncNameLength changes
  if (candidate_name.size() > kMaxFuncNameLength) {
    std::stringstream truncated_name;
    truncated_name << candidate_name.substr(0, kMaxFuncNameLength);
    truncated_name << "_" << std::hash<std::string>{}(candidate_name) << "_";
    candidate_name = truncated_name.str();
  }

  // NB(@jroesch): unfortunately the graph runtime deals with copy in
  // a totally hacky way, we really need to rectify this but this will
  // have to work for now.
  std::string prim_fn_name = candidate_name;
  if (prim_fn_name != "__copy") {
    prim_fn_name = renamer(prim_fn_name);
  }
  auto prim_fn_var = GlobalVar(prim_fn_name);
  prim_fn_var->checked_type_ = prim_func->checked_type();

  // Fusion over tupled results may leave identity relationships
  // between inputs and outputs, and those should not be scheduled.
  // Hence schedule only non PlaceholderOp outputs.
  tvm::Array<te::Tensor> tensor_outs;
  for (const auto& tensor : outputs) {
    if (!tensor->op.as<te::PlaceholderOpNode>()) {
      tensor_outs.push_back(tensor);
    }
  }

  te::Schedule schedule;
  // No need to register schedule for device copy op.
  if (anchor_attrs_.as<DeviceCopyAttrs>() == nullptr && create_schedule_) {
    if (use_auto_scheduler_) {
      const auto* fauto_schedule =
          runtime::Registry::Get("auto_scheduler.relay_integration.auto_schedule_topi_compute");
      ICHECK(fauto_schedule != nullptr)
          << "auto_scheduler.relay_integration.auto_schedule_topi_compute is not registered";
      ObjectRef obj = (*fauto_schedule)(prim_fn_name, tensor_outs);
      if (obj.defined()) {
        schedule = Downcast<te::Schedule>(obj);
      }
    }

    // Use TOPI schdule if user specificed, or the function has no auto_scheduler schedule.
    if (!schedule.defined()) {
      ICHECK(anchor_implementation_.defined());
      schedule = anchor_implementation_.Schedule(anchor_attrs_, tensor_outs, target_);
    }
    for (const auto& scalar : scalars_) {
      if (schedule->Contain(scalar)) {
        schedule[scalar].compute_inline();
      }
    }
  }

  return CachedFunc(target_, prim_fn_var, fn_inputs, outputs, schedule, {});
}
```
### 这部分为返回底层的`ComputeOp`

这句话`auto outputs = this->VisitExpr(prim_func->body);`返回的就是Array<Tensor>，
也是从记录了底层的ComputeOp的地方。
之后调到了`/home2/xiachunwei/Software/tvm/src/relay/backend/te_compiler_cache.cc:260`

```C++
Array<te::Tensor> VisitExpr_(const CallNode* call_node) final {
  static auto fpattern = Op::GetAttrMap<TOpPattern>("TOpPattern");
  static auto flower_call = tvm::runtime::Registry::Get("relay.backend.lower_call");
  ICHECK(flower_call) << "relay.backend.lower_call is not registered.";

  Array<te::Tensor> inputs;
  int count_tuple = 0;
  for (Expr arg : call_node->args) {
    if (arg->checked_type().as<TupleTypeNode>()) {
      ++count_tuple;
    }
    for (te::Tensor tensor : VisitExpr(arg)) {
      inputs.push_back(tensor);
    }
  }

  if (count_tuple) {
    ICHECK_EQ(call_node->args.size(), 1U)
        << "Only functions with a single tuple input are allowed, but " << count_tuple
        << " were provided.";
  }

  ICHECK(call_node->op.as<OpNode>()) << "Primitive function only allows call into primitive ops";
  Op op = Downcast<Op>(call_node->op);

  Array<te::Tensor> outputs;
  OpImplementation impl;
  // Skip fcompute for device copy operators as it is not registered.
  if (op == device_copy_op_) {
    const auto* copy_input = inputs[0].operator->();
    outputs.push_back(te::Tensor(copy_input->shape, copy_input->dtype, te::Operation(), 0));
  } else {
    LoweredOutput lowered_out = (*flower_call)(GetRef<Call>(call_node), inputs, target_);
    outputs = lowered_out->outputs;
    impl = lowered_out->implementation;
  }

  if (create_schedule_) {
    int op_pattern = fpattern[op];
    if (!use_auto_scheduler_ && op_pattern >= kCommReduce) {
      ICHECK(!anchor_op_.defined() || anchor_op_pattern_ < kCommReduce)
          << "Cannot apply TOPI schedule to a primitive function with two complicated ops"
          << " anchor=" << anchor_op_ << " current=" << op;
    }
    if (op_pattern >= anchor_op_pattern_) {
      anchor_op_ = op;
      anchor_attrs_ = call_node->attrs;
      anchor_op_pattern_ = op_pattern;
      anchor_implementation_ = impl;
    }
  }
  if (outputs.size() != 1) {
    const auto* tuple_type = call_node->checked_type().as<TupleTypeNode>();
    ICHECK(tuple_type) << "Expected output to be a tuple type "
                        << PrettyPrint(call_node->checked_type());

    ICHECK_EQ(tuple_type->fields.size(), outputs.size());
  }
  // Set the name to `__copy`. It will be detected in graph runtime to perform
  // data copy across devices.
  if (op == device_copy_op_) {
    readable_name_stream_.str(std::string());
    readable_name_stream_ << "__copy";
  } else {
    readable_name_stream_ << '_' << op->name;
  }
  return outputs;
}
```
之后调用的是这句话`LoweredOutput lowered_out = (*flower_call)(GetRef<Call>(call_node), inputs, target_);`

lower_call

然后调到了`/home2/xiachunwei/Software/tvm/src/relay/ir/op_strategy.cc:80`
```C++
TVM_REGISTER_GLOBAL("relay.op._OpImplementationCompute")
  .set_body([](TVMArgs args, TVMRetValue* rv) {
    OpImplementation imp = args[0];
    Attrs attrs = args[1];
    Array<te::Tensor> inputs = args[2];
    Type out_type = args[3];
    *rv = imp.Compute(attrs, inputs, out_type);
  });
```
然后往下方调用：
```C++
Array<te::Tensor> OpImplementation::Compute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                            const Type& out_type) {
  return (*this)->fcompute(attrs, inputs, out_type);
}
```
然后调用到了relay中具体的op，到`topi::concatenate`中，
```C++
Array<te::Tensor> ConcatenateCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                     const Type& out_type) {
  const ConcatenateAttrs* param = attrs.as<ConcatenateAttrs>();
  ICHECK(param != nullptr);
  return {topi::concatenate(inputs, param->axis)};
}
```
然后是`topi`中的concatenate:
```C++
inline Tensor concatenate(const Array<Tensor>& inputs, int axis = 0, std::string name = "T_concat",
                          std::string tag = kInjective) {
  int ndim = static_cast<int>(inputs[0]->shape.size());
  ICHECK(-ndim <= axis && axis < ndim) << "concatenate only accepts `axis` in [-ndim, ndim)"
                                       << ", but got axis = " << axis << ", and ndim = " << ndim;
  if (axis < 0) {
    axis += ndim;
  }
  ICHECK_LT(axis, inputs[0]->shape.size()) << "axis out of bounds";

  Array<PrimExpr> axis_sizes;
  for (auto t : inputs) {
    axis_sizes.push_back(t->shape[axis]);
  }
  arith::Analyzer analyzer;
  PrimExpr join_size = axis_sizes[0];
  for (size_t i = 1; i < axis_sizes.size(); ++i) {
    join_size += axis_sizes[i];
  }
  join_size = analyzer.Simplify(join_size);
  Array<PrimExpr> out_shape;
  for (size_t i = 0; i < inputs[0]->shape.size(); ++i) {
    out_shape.push_back(i == static_cast<size_t>(axis) ? join_size : inputs[0]->shape[i]);
  }

  return compute(
      out_shape,
      [&](const Array<Var>& indices) {
        auto ret = inputs[0](indices);
        auto ind = indices[axis];
        for (size_t i = 0; i < inputs.size() - 1; ++i) {
          ind -= axis_sizes[i];

          Array<PrimExpr> idx;
          for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
            idx.push_back(indices[i]);
          }
          idx.push_back(ind);
          for (size_t i = axis + 1; i < indices.size(); ++i) {
            idx.push_back(indices[i]);
          }

          ret = tvm::if_then_else(ind >= 0, inputs[i + 1](idx), ret);
        }
        return ret;
      },
      name, tag);
}
```
从以上的调用关系就能够整个得到了整个lower的链条。后续根据返回来的tensor访问底层的op。
既然后续他能够搞schedule，那我也是一样可以访问的。

### 下面这部分是从为PrimFunc做auto-schedule的
在这个schedule里面应该是可以访问其计算的各个轴的。从这个地方跟下去。
`python/tvm/auto_scheduler/relay_integration.py`在python中做的auto_schedule
```C++
@tvm._ffi.register_func("auto_scheduler.relay_integration.auto_schedule_topi_compute")
def auto_schedule_topi(func_name, outs):
    """Use auto-scheduler to schedule any topi compute function.

    Note: This is used internally for relay integration. Do
    not use this as a general user-facing API.

    Parameters
    ----------
    func_name: str
        The name of the function being scheduled.

    outs: List[Tensor]
        The output tensors of topi compute functions

    Returns
    -------
    sch: Optional[te.Schedule]
        A tuned schedule or none (if not tuned) in the final build mode;
        None in the tracing mode so that the fallback topi schedule will be used.
    """

    # pylint: disable=import-outside-toplevel
    from tvm.auto_scheduler.measure import (
        prepare_input_map,
    )  # lazily import to avoid recursive dependency

    io_tensors, has_layout_free, has_complex_op = traverse_to_get_io_tensors(outs)
    if not io_tensors:  # The compute includes dynamic shapes which are not supported yet.
        return None

    try:
        dag = ComputeDAG(io_tensors)
    except tvm.error.TVMError as err:
        logger.info("Failed to create a ComputeDAG for auto_scheduler: %s", str(err))
        return None

    key = register_workload_tensors(dag.workload_key(), io_tensors)
    target = tvm.target.Target.current()

    dispatch_ctx = DispatchContext.current
    state = dispatch_ctx.query(target, key, has_complex_op, dag, func_name)
    schedule = None

    env = TracingEnvironment.current
    if env is None:
        # in the final build mode
        if state is None:
            return None

        schedule, _ = dag.apply_steps_from_state(state)
        return schedule

    if env.tracing_mode in [TracingMode.EXTRACT_TASK, TracingMode.EXTRACT_COMPLEX_TASK_ONLY]:
        # in the task extraction mode
        if has_complex_op or env.tracing_mode == TracingMode.EXTRACT_TASK:
            env.add_workload_key(func_name, key)
            input_map = prepare_input_map(io_tensors)
            if input_map:
                env.add_workload_input_names(key, list(input_map.values()))
    elif env.tracing_mode == TracingMode.PREPARE_LAYOUT_REWRITE:
        # in prepare_layout_rewrite mode
        if (
            LayoutRewriteOption.get_target_default(target, True) != LayoutRewriteOption.NO_REWRITE
            and has_layout_free
        ):
            if state is None:
                return None

            # rewrite the layout and update the context for the new dag
            new_dag = dag.rewrite_layout_from_state(state)
            new_key = new_dag.workload_key()
            if new_key != key:
                dispatch_ctx.update(target, new_key, state)
    else:
        raise ValueError("Invalid tracing mode: " + env.tracing_mode)

    return schedule
```

之后`src/auto_scheduler/compute_dag.cc`的自动调度中的ComputeDAG::ApplySteps()是主要函数：
获取子图或者一个TIR的input和output tensor，然后根据output tensor创建schedule，再为每个output 的op创建State。
从这里开始调用`te::create_schedule`就和之前的te中create_schedule的部分连接起来了。
```C++
std::pair<te::Schedule, Array<te::Tensor>> ComputeDAG::ApplySteps(
    const Array<Step>& transform_steps, Array<te::Stage>* stages, StageToAxesMap* stage_to_axes,
    LayoutRewriteOption layout_rewrite) const {
  if (layout_rewrite != LayoutRewriteOption::NoRewrite && HasLayoutFreeTensors(*this) &&
      !transform_steps.empty()) {
    Array<Step> steps = transform_steps;
    const auto& dag = RewriteLayout(&steps, layout_rewrite);
    return dag.ApplySteps(steps);
  }

  // Temporal object to be used if the input pointer is nullptr
  Array<te::Stage> temp_stages;
  StageToAxesMap temp_stage_to_axes;
  if (stages == nullptr) {
    stages = &temp_stages;
  }
  if (stage_to_axes == nullptr) {
    stage_to_axes = &temp_stage_to_axes;
  }
  Array<te::Operation> out_ops;
  for (const auto& op : operator->()->ops) {
    if (operator->()->access_analyzer.IsOutput(op)) {
      out_ops.push_back(op);
    }
  }

  // Create the initial schedule
  te::Schedule schedule = te::create_schedule(out_ops);

  // init axes
  for (const auto& x : operator->()->ops) {
    const te::Stage& stage = schedule[x];
    stages->push_back(stage);
    UpdateStageToAxesMap(stage, stage_to_axes);
  }

  // Apply the history steps to TVM schedule
  // Call each step's ApplyToSchedule method
  for (const auto& step : transform_steps) {
    StepApplyToSchedule(step, stages, stage_to_axes, &schedule, transform_steps);
  }

  return std::make_pair(schedule, operator->()->tensors);
}
```

把op搞成te::ComputeOpNode，并且访问其axis
```C++
// Update the te::stage to tir::IterVar axis mapping
void UpdateStageToAxesMap(const te::Stage& stage, StageToAxesMap* stage_to_axes) {
  if (auto pop = stage->op.as<te::ComputeOpNode>()) {
    Array<IterVar> axes;
    for (const auto& axis : pop->axis) {
      axes.push_back(axis);
    }
    for (const auto& axis : pop->reduce_axis) {
      axes.push_back(axis);
    }
    stage_to_axes->Set(stage, std::move(axes));
  } else if (stage->op->IsInstance<te::PlaceholderOpNode>()) {
    {}  // do nothing on Placeholder
  } else {
    LOG(FATAL) << "Invalid op " << stage->op;
  }
}
```
如何`StepApplyToSchedule`:
```C++
void StepApplyToSchedule(const Step& step, Array<te::Stage>* stages, StageToAxesMap* stage_to_axes,
                         te::Schedule* schedule, const Array<Step>& transform_steps) {
  if (auto ps = step.as<AnnotationStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes);
  } else if (auto ps = step.as<FuseStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes);
  } else if (auto ps = step.as<PragmaStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes);
  } else if (auto ps = step.as<ReorderStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes);
  } else if (auto ps = step.as<SplitStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes);
  } else if (auto ps = step.as<FollowSplitStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes, transform_steps);
  } else if (auto ps = step.as<FollowFusedSplitStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes, transform_steps);
  } else if (auto ps = step.as<StorageAlignStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes);
  } else if (auto ps = step.as<ComputeAtStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes);
  } else if (auto ps = step.as<ComputeInlineStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes);
  } else if (auto ps = step.as<ComputeRootStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes);
  } else if (auto ps = step.as<CacheReadStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes, schedule);
  } else if (auto ps = step.as<CacheWriteStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes, schedule);
  } else if (auto ps = step.as<RfactorStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes, schedule);
  } else {
    LOG(FATAL) << "Invalid Step: " << step;
  }
}
```
保存到一个map里面
```C++
void AnnotationStepNode::ApplyToSchedule(Array<te::Stage>* stages,
                                         StageToAxesMap* stage_to_axes) const {
  te::Stage stage = (*stages)[stage_id];
  const Array<IterVar>& axes = (*stage_to_axes)[stage];

  switch (annotation) {
    case IteratorAnnotation::kUnroll:
      stage.unroll(axes[iter_id]);
      break;
    case IteratorAnnotation::kVectorize:
      stage.vectorize(axes[iter_id]);
      break;
    case IteratorAnnotation::kParallel:
      stage.parallel(axes[iter_id]);
      break;
    case IteratorAnnotation::kVThread:
    case IteratorAnnotation::kBlockX:
    case IteratorAnnotation::kBlockY:
    case IteratorAnnotation::kBlockZ:
    case IteratorAnnotation::kThreadX:
    case IteratorAnnotation::kThreadY:
    case IteratorAnnotation::kThreadZ:
      stage.bind(axes[iter_id],
                 te::thread_axis(Range(), IteratorAnnotationString[static_cast<int>(annotation)]));
      break;
    case IteratorAnnotation::kNone:
      break;
    default:
      LOG(FATAL) << "Invalid Annotation " << static_cast<int>(annotation);
      break;
  }

  stages->Set(stage_id, std::move(stage));
}
```


### 如何获取op的tensortype
在`src/relay/backend/graph_plan_memory.cc`，`op->checked_type()`获取relay::type类型，其子类包含`TensorType`，其中包括Tensor的shape等。
```C++
void CreateTokenOnDevice(const ExprNode* op, DLDeviceType device_type,
                          bool can_realloc) override {
  ICHECK(!token_map_.count(op));
  std::vector<StorageToken*> tokens;
  for (const auto& ttype : FlattenTupleType(op->checked_type())) {
    StorageToken* token = arena_->make<StorageToken>();
    token->ttype = ttype;
    // TODO(mbs): Should be TargetDevice.
    token->device.device_type = device_type;
    token->device.device_id = 0;
    tokens.push_back(token);
  }
  token_map_[op] = tokens;
}
```
`tests/cpp/relay_build_module_test.cc`中有用C++的API build一个compute并且run起来的完整示例。

### grep出来的TIR

```C++
function.cc:82:TVM_REGISTER_GLOBAL("tir.PrimFunc")
stmt_functor.cc:744:TVM_REGISTER_GLOBAL("tir.IRTransform").set_body_typed(IRTransform);
stmt_functor.cc:746:TVM_REGISTER_GLOBAL("tir.PostOrderVisit").set_body_typed([](ObjectRef node, PackedFunc f) {
stmt_functor.cc:750:TVM_REGISTER_GLOBAL("tir.Substitute")
expr.cc:105:TVM_REGISTER_GLOBAL("tir.Var").set_body_typed([](String name_hint, runtime::TVMArgValue type,
expr.cc:133:TVM_REGISTER_GLOBAL("tir.SizeVar").set_body_typed([](String s, DataType t, Span span) {
expr.cc:156:TVM_REGISTER_GLOBAL("tir.IterVar")
expr.cc:188:TVM_REGISTER_GLOBAL("tir.StringImm").set_body_typed([](String value, Span span) {
expr.cc:211:TVM_REGISTER_GLOBAL("tir.Cast").set_body_typed([](DataType dtype, PrimExpr value, Span span) {
expr.cc:228:TVM_REGISTER_GLOBAL("tir.Add").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
expr.cc:247:TVM_REGISTER_GLOBAL("tir.Sub").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
expr.cc:266:TVM_REGISTER_GLOBAL("tir.Mul").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
expr.cc:285:TVM_REGISTER_GLOBAL("tir.Div").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
expr.cc:304:TVM_REGISTER_GLOBAL("tir.Mod").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
expr.cc:323:TVM_REGISTER_GLOBAL("tir.FloorDiv").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
expr.cc:338:TVM_REGISTER_GLOBAL("tir.FloorMod").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
expr.cc:353:TVM_REGISTER_GLOBAL("tir.Min").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
expr.cc:372:TVM_REGISTER_GLOBAL("tir.Max").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
expr.cc:391:TVM_REGISTER_GLOBAL("tir.EQ").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
expr.cc:410:TVM_REGISTER_GLOBAL("tir.NE").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
expr.cc:429:TVM_REGISTER_GLOBAL("tir.LT").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
expr.cc:448:TVM_REGISTER_GLOBAL("tir.LE").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
expr.cc:467:TVM_REGISTER_GLOBAL("tir.GT").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
expr.cc:486:TVM_REGISTER_GLOBAL("tir.GE").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
expr.cc:518:TVM_REGISTER_GLOBAL("tir.And").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
expr.cc:550:TVM_REGISTER_GLOBAL("tir.Or").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
expr.cc:578:TVM_REGISTER_GLOBAL("tir.Not").set_body_typed([](PrimExpr a, Span span) { return Not(a, span); });
expr.cc:607:TVM_REGISTER_GLOBAL("tir.Select")
expr.cc:678:TVM_REGISTER_GLOBAL("tir.Load").set_body([](TVMArgs args, TVMRetValue* ret) {
expr.cc:721:TVM_REGISTER_GLOBAL("tir.Ramp")
expr.cc:752:TVM_REGISTER_GLOBAL("tir.Broadcast").set_body_typed([](PrimExpr value, int lanes, Span span) {
expr.cc:781:TVM_REGISTER_GLOBAL("tir.Let").set_body_typed([](Var var, PrimExpr value, PrimExpr body,
expr.cc:812:TVM_REGISTER_GLOBAL("tir.Call")
expr.cc:889:TVM_REGISTER_GLOBAL("tir.Shuffle")
expr.cc:971:TVM_REGISTER_GLOBAL("tir.CommReducer")
expr.cc:977:TVM_REGISTER_GLOBAL("tir.CommReducerCombine")
expr.cc:1022:TVM_REGISTER_GLOBAL("tir.Reduce")
expr.cc:1050:TVM_REGISTER_GLOBAL("tir.Any").set_body_typed([](Span span) { return Any(span); });
expr.cc:1067:TVM_REGISTER_GLOBAL("tir.BufferLoad")
expr.cc:1097:TVM_REGISTER_GLOBAL("tir.ProducerLoad")
data_layout.cc:366:TVM_REGISTER_GLOBAL("tir.Layout").set_body_typed([](std::string name) { return Layout(name); });
data_layout.cc:368:TVM_REGISTER_GLOBAL("tir.LayoutIndexOf").set_body_typed([](Layout layout, std::string axis) -> int {
data_layout.cc:372:TVM_REGISTER_GLOBAL("tir.LayoutFactorOf")
data_layout.cc:377:TVM_REGISTER_GLOBAL("tir.LayoutNdim").set_body_typed([](Layout layout) -> int {
data_layout.cc:381:TVM_REGISTER_GLOBAL("tir.LayoutGetItem").set_body_typed([](Layout layout, int idx) -> std::string {
data_layout.cc:386:TVM_REGISTER_GLOBAL("tir.BijectiveLayout")
data_layout.cc:391:TVM_REGISTER_GLOBAL("tir.BijectiveLayoutForwardIndex")
data_layout.cc:394:TVM_REGISTER_GLOBAL("tir.BijectiveLayoutBackwardIndex")
data_layout.cc:397:TVM_REGISTER_GLOBAL("tir.BijectiveLayoutForwardShape")
data_layout.cc:400:TVM_REGISTER_GLOBAL("tir.BijectiveLayoutBackwardShape")
stmt.cc:48:TVM_REGISTER_GLOBAL("tir.LetStmt")
stmt.cc:76:TVM_REGISTER_GLOBAL("tir.AttrStmt")
stmt.cc:111:TVM_REGISTER_GLOBAL("tir.AssertStmt")
stmt.cc:155:TVM_REGISTER_GLOBAL("tir.For").set_body_typed(
stmt.cc:217:TVM_REGISTER_GLOBAL("tir.While").set_body_typed([](PrimExpr condition, Stmt body, Span span) {
stmt.cc:273:TVM_REGISTER_GLOBAL("tir.Store").set_body([](TVMArgs args, TVMRetValue* ret) {
stmt.cc:312:TVM_REGISTER_GLOBAL("tir.ProducerStore")
stmt.cc:377:TVM_REGISTER_GLOBAL("tir.Allocate")
stmt.cc:428:TVM_REGISTER_GLOBAL("tir.ProducerRealize")
stmt.cc:469:TVM_REGISTER_GLOBAL("tir.Prefetch")
stmt.cc:500:TVM_REGISTER_GLOBAL("tir.SeqStmt").set_body_typed([](Array<Stmt> seq, Span span) {
stmt.cc:529:TVM_REGISTER_GLOBAL("tir.IfThenElse")
stmt.cc:575:TVM_REGISTER_GLOBAL("tir.Evaluate").set_body_typed([](PrimExpr value, Span span) {
stmt.cc:599:TVM_REGISTER_GLOBAL("tir.BufferStore")
stmt.cc:627:TVM_REGISTER_GLOBAL("tir.BufferRealize")
stmt.cc:688:TVM_REGISTER_GLOBAL("tir.BufferRegion").set_body_typed([](Buffer buffer, Array<Range> region) {
stmt.cc:762:TVM_REGISTER_GLOBAL("tir.MatchBufferRegion").set_body_typed([](Buffer buffer, BufferRegion source) {
stmt.cc:796:TVM_REGISTER_GLOBAL("tir.Block")
stmt.cc:892:TVM_REGISTER_GLOBAL("tir.BlockRealize")
buffer.cc:457:TVM_REGISTER_GLOBAL("tir.Buffer").set_body([](TVMArgs args, TVMRetValue* ret) {
buffer.cc:465:TVM_REGISTER_GLOBAL("tir.BufferAccessPtr").set_body_method(&Buffer::access_ptr);
buffer.cc:467:TVM_REGISTER_GLOBAL("tir.BufferVLoad").set_body_method(&Buffer::vload);
buffer.cc:469:TVM_REGISTER_GLOBAL("tir.BufferVStore").set_body_method(&Buffer::vstore);
buffer.cc:471:TVM_REGISTER_GLOBAL("tir.BufferStorageScope").set_body_method(&Buffer::scope);
specialize.cc:375:TVM_REGISTER_GLOBAL("tir.Specialize").set_body_typed(Specialize);
transform.cc:125:TVM_REGISTER_GLOBAL("tir.transform.CreatePrimFuncPass")
script/script_complete.cc:128:TVM_REGISTER_GLOBAL("script.Complete").set_body_typed(ScriptComplete);
```

### TVM中的relay的passes
```python
DynamicToStatic:
EliminateCommonSubexpr:
SimplifyExpr:
CombineParallelConv2d:
CombineParallelDense:
CombineParallelBatchMatmul:
FoldConstant:
BackwardFoldScaleAxis:
ForwardFoldScaleAxis:
FoldConstant:
CanonicalizeCast:
CanonicalizeOps:
AlterOpLayout:
FastMath:
FoldConstant:
SplitArgs:
PlanDevicesRewrite:
HorizontalFusion:
FuseOps:
LabelOps:
GraphExecutorCodegen:
```

### 把output_tensors传递出来
```C++
[this](Function func) {
      // We need to maintain the constant map for external
      // functions so we pass this processing function which
      // allows us to process each function as we lower it.
      if (func->GetAttr<String>(attr::kCompiler).defined()) {
        UpdateConstants(func, &params_);
      }

      // TODO(@areusch, @jroesch): We should refactor this to
      // execute as a further pass, instead writing data to the
      // lowering process directly.
      tec::UpdateFunctionMetadata(func, this->function_metadata_);
    }
```