# TVM typical function call stack

### `LowerInternal`

`LowerInternal`函数调用栈如下：
```C++
#0  tvm::relay::tec::TECompilerImpl::LowerInternal(tvm::relay::tec::CCacheKey const&, std::function<tvm::runtime::String (tvm::runtime::String)>) (this=0x55555614dd70, key=..., mangle_fn=...)
    at /home2/xiachunwei/Software/tvm/src/relay/backend/te_compiler.cc:200
#1  0x00007ffea07a1a91 in tvm::relay::tec::TECompilerImpl::Lower(tvm::relay::tec::CCacheKey const&, std::function<tvm::runtime::String (tvm::runtime::String)>) (this=0x55555614dd70, key=..., 
    mangle_fn=...) at /home2/xiachunwei/Software/tvm/src/relay/backend/te_compiler.cc:65
#2  0x00007ffea07a1c56 in tvm::relay::tec::TECompilerImpl::Lower (this=0x55555614dd70, key=..., mod_name=...) at /home2/xiachunwei/Software/tvm/src/relay/backend/te_compiler.cc:71
#3  0x00007ffea07a6889 in tvm::relay::tec::LowerTensorExprMutator::LowerFunction (this=0x7fffffff97b0, func=..., target=...) at /home2/xiachunwei/Software/tvm/src/relay/backend/te_compiler.cc:484
#4  0x00007ffea07a8d20 in tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_ (this=0x7fffffff97b0, call_node=0x555556ccd3b0)
    at /home2/xiachunwei/Software/tvm/src/relay/backend/te_compiler.cc:613
#5  0x00007ffea055a987 in tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_ (this=0x7fffffff97b0, call_node=0x555556ccd3b0)
    at /home2/xiachunwei/Software/tvm/src/relay/transforms/device_aware_visitors.cc:267
#6  0x00007ffe9f72fadd in tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::InitVTable()::{lambda(tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>*)#6}::operator()(tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>*) const (__closure=0x0, n=..., self=0x7fffffff97b0)
    at /home2/xiachunwei/Software/tvm/include/tvm/relay/expr_functor.h:128
#7  0x00007ffe9f72fb38 in tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::InitVTable()::{lambda(tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>*)#6}::_FUN(tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>*) () at /home2/xiachunwei/Software/tvm/include/tvm/relay/expr_functor.h:128
#8  0x00007ffe9f7306ce in tvm::NodeFunctor<tvm::RelayExpr (tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>*)>::operator()(tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>*) const (
    this=0x7ffea50e3e00 <tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)::vtable>, n=..., args#0=0x7fffffff97b0)
    at /home2/xiachunwei/Software/tvm/include/tvm/node/functor.h:97
#9  0x00007ffe9f72ea44 in tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&) (this=0x7fffffff97b0, n=...)
    at /home2/xiachunwei/Software/tvm/include/tvm/relay/expr_functor.h:95
#10 0x00007ffea08783b4 in tvm::relay::ExprMutator::VisitExpr (this=0x7fffffff97b0, expr=...) at /home2/xiachunwei/Software/tvm/src/relay/ir/expr_functor.cc:156
#11 0x00007ffea04c8428 in tvm::relay::ExprMutator::Mutate (this=0x7fffffff97b0, expr=...) at /home2/xiachunwei/Software/tvm/include/tvm/relay/expr_functor.h:190
#12 0x00007ffea0878bb4 in tvm::relay::ExprMutator::VisitExpr_ (this=0x7fffffff97b0, op=0x555556cb1260) at /home2/xiachunwei/Software/tvm/src/relay/ir/expr_functor.cc:214
#13 0x00007ffea055aa2c in tvm::relay::transform::DeviceAwareExprMutator::DeviceAwareVisitExpr_ (this=0x7fffffff97b0, function_node=0x555556cb1260)
    at /home2/xiachunwei/Software/tvm/src/relay/transforms/device_aware_visitors.cc:272
#14 0x00007ffea055a26a in tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_ (this=0x7fffffff97b0, function_node=0x555556cb1260)
    at /home2/xiachunwei/Software/tvm/src/relay/transforms/device_aware_visitors.cc:217
#15 0x00007ffe9f72fa01 in tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::InitVTable()::{lambda(tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>*)#5}::operator()(tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>*) const (__closure=0x0, n=..., self=0x7fffffff97b0)
    at /home2/xiachunwei/Software/tvm/include/tvm/relay/expr_functor.h:127
#16 0x00007ffe9f72fa5c in tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::InitVTable()::{lambda(tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>*)#5}::_FUN(tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>*) () at /home2/xiachunwei/Software/tvm/include/tvm/relay/expr_functor.h:127
#17 0x00007ffe9f7306ce in tvm::NodeFunctor<tvm::RelayExpr (tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>*)>::operator()(tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>*) const (
    this=0x7ffea50e3e00 <tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)::vtable>, n=..., args#0=0x7fffffff97b0)
    at /home2/xiachunwei/Software/tvm/include/tvm/node/functor.h:97
---Type <return> to continue, or q <return> to quit---
#18 0x00007ffe9f72ea44 in tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&) (this=0x7fffffff97b0, n=...)
    at /home2/xiachunwei/Software/tvm/include/tvm/relay/expr_functor.h:95
#19 0x00007ffea08783b4 in tvm::relay::ExprMutator::VisitExpr (this=0x7fffffff97b0, expr=...) at /home2/xiachunwei/Software/tvm/src/relay/ir/expr_functor.cc:156
#20 0x00007ffea04c8428 in tvm::relay::ExprMutator::Mutate (this=0x7fffffff97b0, expr=...) at /home2/xiachunwei/Software/tvm/include/tvm/relay/expr_functor.h:190
#21 0x00007ffea0794744 in tvm::relay::tec::<lambda(tvm::relay::Function, tvm::IRModule, tvm::relay::transform::PassContext)>::operator()(tvm::relay::Function, tvm::IRModule, tvm::relay::transform::PassContext) const (__closure=0x555556cb72a0, func=..., module=..., ctx=...) at /home2/xiachunwei/Software/tvm/src/relay/backend/te_compiler.cc:678
#22 0x00007ffea079ade0 in tvm::runtime::detail::unpack_call_dispatcher<tvm::relay::Function, 0, 3, tvm::relay::tec::LowerTensorExpr(tvm::relay::tec::TargetMap, const tvm::runtime::String&, tvm::relay::tec::TECompiler, std::function<void(tvm::relay::Function)>)::<lambda(tvm::relay::Function, tvm::IRModule, tvm::relay::transform::PassContext)> >::run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_> (rv=0x7fffffff9d70, args_pack=..., f=..., optional_name=0x0)
    at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1397
#23 tvm::runtime::detail::unpack_call_dispatcher<tvm::relay::Function, 1, 2, tvm::relay::tec::LowerTensorExpr(tvm::relay::tec::TargetMap, const tvm::runtime::String&, tvm::relay::tec::TECompiler, std::function<void(tvm::relay::Function)>)::<lambda(tvm::relay::Function, tvm::IRModule, tvm::relay::transform::PassContext)> >::run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_> (rv=0x7fffffff9d70, args_pack=..., f=..., optional_name=0x0) at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1382
#24 tvm::runtime::detail::unpack_call_dispatcher<tvm::relay::Function, 2, 1, tvm::relay::tec::LowerTensorExpr(tvm::relay::tec::TargetMap, const tvm::runtime::String&, tvm::relay::tec::TECompiler, std::function<void(tvm::relay::Function)>)::<lambda(tvm::relay::Function, tvm::IRModule, tvm::relay::transform::PassContext)> >::run<tvm::runtime::TVMMovableArgValueWithContext_> (rv=0x7fffffff9d70, 
    args_pack=..., f=..., optional_name=0x0) at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1382
#25 tvm::runtime::detail::unpack_call_dispatcher<tvm::relay::Function, 3, 0, tvm::relay::tec::LowerTensorExpr(tvm::relay::tec::TargetMap, const tvm::runtime::String&, tvm::relay::tec::TECompiler, std::function<void(tvm::relay::Function)>)::<lambda(tvm::relay::Function, tvm::IRModule, tvm::relay::transform::PassContext)> >::run<> (rv=0x7fffffff9d70, args_pack=..., f=..., optional_name=0x0)
    at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1382
#26 tvm::runtime::detail::unpack_call<tvm::relay::Function, 3, tvm::relay::tec::LowerTensorExpr(tvm::relay::tec::TargetMap, const tvm::runtime::String&, tvm::relay::tec::TECompiler, std::function<void(tvm::relay::Function)>)::<lambda(tvm::relay::Function, tvm::IRModule, tvm::relay::transform::PassContext)> > (rv=0x7fffffff9d70, args=..., f=..., optional_name=0x0)
    at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1421
#27 tvm::runtime::TypedPackedFunc<tvm::relay::Function(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)>::<lambda(const tvm::runtime::TVMArgs&, tvm::runtime::TVMRetValue*)>::operator()(const tvm::runtime::TVMArgs &, tvm::runtime::TVMRetValue *) const (__closure=0x555556cb72a0, args=..., rv=0x7fffffff9d70)
    at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1492
#28 0x00007ffea079edbb in std::_Function_handler<void(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<R(Args ...)>::AssignTypedLambda(FType) [with FLambda = tvm::relay::tec::LowerTensorExpr(tvm::relay::tec::TargetMap, const tvm::runtime::String&, tvm::relay::tec::TECompiler, std::function<void(tvm::relay::Function)>)::<lambda(tvm::relay::Function, tvm::IRModule, tvm::relay::transform::PassContext)>; R = tvm::relay::Function; Args = {tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext}]::<lambda(const tvm::runtime::TVMArgs&, tvm::runtime::TVMRetValue*)> >::_M_invoke(const std::_Any_data &, tvm::runtime::TVMArgs &&, tvm::runtime::TVMRetValue *&&) (__functor=..., __args#0=..., __args#1=@0x7fffffff9b50: 0x7fffffff9d70)
    at /usr/include/c++/7/bits/std_function.h:316
#29 0x00007ffe9f570cc0 in std::function<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)>::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const (this=0x555556cb3b40, 
    __args#0=..., __args#1=0x7fffffff9d70) at /usr/include/c++/7/bits/std_function.h:706
#30 0x00007ffea08a8ae0 in tvm::runtime::PackedFunc::operator()<tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext> (this=0x555556cb3b40)
    at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1369
#31 0x00007ffea08a60e8 in tvm::runtime::detail::typed_packed_call_dispatcher<tvm::relay::Function>::run<tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext> (pf=...)
    at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1444
#32 tvm::runtime::TypedPackedFunc<tvm::relay::Function (tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)>::operator()(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext) const (args#2=..., args#1=..., args#0=..., this=0x555556cb3b40) at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1498
#33 tvm::relay::transform::FunctionPassNode::operator() (this=0x555556cb3b20, mod=..., pass_ctx=...) at /home2/xiachunwei/Software/tvm/src/relay/ir/transform.cc:144
#34 0x00007ffe9f75b1fc in tvm::transform::Pass::operator() (this=0x7fffffffa400, mod=..., pass_ctx=...) at /home2/xiachunwei/Software/tvm/src/ir/transform.cc:267
#35 0x00007ffe9f75aeae in tvm::transform::Pass::operator() (this=0x7fffffffa400, mod=...) at /home2/xiachunwei/Software/tvm/src/ir/transform.cc:255
#36 0x00007ffea079891b in tvm::relay::tec::LowerTE(tvm::IRModule const&, std::unordered_map<DLDeviceType, tvm::Target, tvm::relay::backend::EnumClassHash, std::equal_to<DLDeviceType>, std::allocator<std::pair<DLDeviceType const, tvm::Target> > >, tvm::runtime::String const&, std::function<void (tvm::relay::Function)>) (module=..., targets=std::unordered_map with 1 element = {...}, 
    module_name=..., process_fn=...) at /home2/xiachunwei/Software/tvm/src/relay/backend/te_compiler.cc:903
#37 0x00007ffea07994be in tvm::relay::tec::<lambda(tvm::IRModule, tvm::relay::transform::PassContext)>::operator()(tvm::IRModule, tvm::relay::transform::PassContext) const (
    __closure=0x555556cb4870, module=..., ctx=...) at /home2/xiachunwei/Software/tvm/src/relay/backend/te_compiler.cc:953
#38 0x00007ffea079b598 in tvm::runtime::detail::unpack_call_dispatcher<tvm::IRModule, 0, 2, tvm::relay::tec::LowerTEPass(tvm::relay::tec::TargetMap, const tvm::runtime::String&, std::function<void(tvm::relay::Function)>)::<lambda(tvm::IRModule, tvm::relay::transform::PassContext)> >::run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_> (
    rv=0x7fffffffa9f0, args_pack=..., f=..., optional_name=0x0) at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1397
#39 tvm::runtime::detail::unpack_call_dispatcher<tvm::IRModule, 1, 1, tvm::relay::tec::LowerTEPass(tvm::relay::tec::TargetMap, const tvm::runtime::String&, std::function<void(tvm::relay::Function)>)::<lambda(tvm::IRModule, tvm::relay::transform::PassContext)> >::run<tvm::runtime::TVMMovableArgValueWithContext_> (rv=0x7fffffffa9f0, args_pack=..., f=..., optional_name=0x0)
    at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1382
#40 tvm::runtime::detail::unpack_call_dispatcher<tvm::IRModule, 2, 0, tvm::relay::tec::LowerTEPass(tvm::relay::tec::TargetMap, const tvm::runtime::String&, std::function<void(tvm::relay::Function)>)::<lambda(tvm::IRModule, tvm::relay::transform::PassContext)> >::run<> (rv=0x7fffffffa9f0, args_pack=..., f=..., optional_name=0x0)
    at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1382
#41 tvm::runtime::detail::unpack_call<tvm::IRModule, 2, tvm::relay::tec::LowerTEPass(tvm::relay::tec::TargetMap, const tvm::runtime::String&, std::function<void(tvm::relay::Function)>)::<lambda(tvm::IRModule, tvm::relay::transform::PassContext)> > (rv=0x7fffffffa9f0, args=..., f=..., optional_name=0x0) at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1421
#42 tvm::runtime::TypedPackedFunc<tvm::IRModule(tvm::IRModule, tvm::transform::PassContext)>::<lambda(const tvm::runtime::TVMArgs&, tvm::runtime::TVMRetValue*)>::operator()(const tvm::runtime::TVMArgs &, tvm::runtime::TVMRetValue *) const (__closure=0x555556cb4870, args=..., rv=0x7fffffffa9f0) at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1492
#43 0x00007ffea079ef56 in std::_Function_handler<void(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<R(Args ...)>::AssignTypedLambda(FType) [with FLambda = tvm::relay::tec::LowerTEPass(tvm::relay::tec::TargetMap, const tvm::runtime::String&, std::function<void(tvm::relay::Function)>)::<lambda(tvm::IRModule, tvm::relay::transform::PassContext)>; R = tvm::IRModule; Args = {tvm::IRModule, tvm::transform::PassContext}]::<lambda(const tvm::runtime::TVMArgs&, tvm::runtime::TVMRetValue*)> >::_M_invoke(const std::_Any_data &, tvm::runtime::TVMArgs &&, tvm::runtime::TVMRetValue *&&) (__functor=..., __args#0=..., __args#1=@0x7fffffffa880: 0x7fffffffa9f0) at /usr/include/c++/7/bits/std_function.h:316
#44 0x00007ffe9f570cc0 in std::function<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)>::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const (this=0x555556caef70, 
    __args#0=..., __args#1=0x7fffffffa9f0) at /usr/include/c++/7/bits/std_function.h:706
---Type <return> to continue, or q <return> to quit---
#45 0x00007ffe9f76bc72 in tvm::runtime::PackedFunc::operator()<tvm::IRModule, tvm::transform::PassContext> (this=0x555556caef70)
    at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1369
#46 0x00007ffe9f75bd36 in tvm::runtime::detail::typed_packed_call_dispatcher<tvm::IRModule>::run<tvm::IRModule, tvm::transform::PassContext> (pf=...)
    at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1444
#47 tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::operator()(tvm::IRModule, tvm::transform::PassContext) const (args#1=..., args#0=..., 
    this=0x555556caef70) at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1498
#48 tvm::transform::ModulePassNode::operator() (this=0x555556caef50, mod=..., pass_ctx=...) at /home2/xiachunwei/Software/tvm/src/ir/transform.cc:416
#49 0x00007ffe9f75b1fc in tvm::transform::Pass::operator() (this=0x7fffffffafe8, mod=..., pass_ctx=...) at /home2/xiachunwei/Software/tvm/src/ir/transform.cc:267
#50 0x00007ffe9f75d051 in tvm::transform::SequentialNode::operator() (this=0x555556cb50f0, mod=..., pass_ctx=...) at /home2/xiachunwei/Software/tvm/src/ir/transform.cc:487
#51 0x00007ffe9f75b1fc in tvm::transform::Pass::operator() (this=0x7fffffffb570, mod=..., pass_ctx=...) at /home2/xiachunwei/Software/tvm/src/ir/transform.cc:267
#52 0x00007ffe9f75aeae in tvm::transform::Pass::operator() (this=0x7fffffffb570, mod=...) at /home2/xiachunwei/Software/tvm/src/ir/transform.cc:255
#53 0x00007ffea07341be in tvm::relay::backend::GraphExecutorCodegen::Codegen (this=0x555556b4c110, func=..., mod_name=...)
    at /home2/xiachunwei/Software/tvm/src/relay/backend/graph_executor_codegen.cc:238
#54 0x00007ffea0739edf in tvm::relay::backend::GraphExecutorCodegenModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#2}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const (__closure=0x555556b26ea0, 
    args=..., rv=0x7fffffffbc00) at /home2/xiachunwei/Software/tvm/src/relay/backend/graph_executor_codegen.cc:636
#55 0x00007ffea07452f5 in std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::relay::backend::GraphExecutorCodegenModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#2}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&) (__functor=..., __args#0=..., __args#1=@0x7fffffffbb00: 0x7fffffffbc00) at /usr/include/c++/7/bits/std_function.h:316
#56 0x00007ffe9f570cc0 in std::function<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)>::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const (this=0x7fffffffbc10, 
    __args#0=..., __args#1=0x7fffffffbc00) at /usr/include/c++/7/bits/std_function.h:706
#57 0x00007ffea071fc1c in tvm::runtime::PackedFunc::operator()<tvm::relay::Function, tvm::runtime::String> (this=0x7fffffffbc10)
    at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1369
#58 0x00007ffea071dc0e in tvm::relay::backend::ExecutorCodegen::CallFunc<tvm::relay::Function, tvm::runtime::String> (this=0x555556cccd10, name="codegen")
    at /home2/xiachunwei/Software/tvm/src/relay/backend/build_module.cc:112
#59 0x00007ffea07168ff in tvm::relay::backend::ExecutorCodegen::Codegen (this=0x555556cccd10, func=..., mod_name=...) at /home2/xiachunwei/Software/tvm/src/relay/backend/build_module.cc:61
#60 0x00007ffea071c09c in tvm::relay::backend::RelayBuildModule::BuildRelay (this=0x55555621bc40, relay_module=..., params=std::unordered_map with 0 elements, mod_name=...)
    at /home2/xiachunwei/Software/tvm/src/relay/backend/build_module.cc:459
#61 0x00007ffea0719c75 in tvm::relay::backend::RelayBuildModule::Build (this=0x55555621bc40, mod=..., targets=..., target_host=..., executor=..., mod_name=...)
    at /home2/xiachunwei/Software/tvm/src/relay/backend/build_module.cc:297
#62 0x00007ffea0717cb5 in tvm::relay::backend::RelayBuildModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#3}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const (__closure=0x555556aaacc0, args=..., 
    rv=0x7fffffffc340) at /home2/xiachunwei/Software/tvm/src/relay/backend/build_module.cc:181
```


### `te::compute`

```C++
#0  tvm::te::compute(tvm::runtime::Array<tvm::PrimExpr, void>, std::function<tvm::PrimExpr (tvm::runtime::Array<tvm::tir::Var, void> const&)>, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, tvm::runtime::Map<tvm::runtime::String, tvm::runtime::ObjectRef, void, void>) (shape=..., fcompute=..., 
    name="T_concat", tag="injective", attrs=...) at /home2/xiachunwei/Software/tvm/src/te/operation/compute_op.cc:95
#1  0x00007ffe9ff4fac8 in tvm::topi::concatenate (inputs=..., axis=1, name="T_concat", tag="injective")
    at /home2/xiachunwei/Software/tvm/include/tvm/topi/transform.h:430
#2  0x00007ffea030a8e0 in tvm::relay::ConcatenateCompute (attrs=..., inputs=..., out_type=...)
    at /home2/xiachunwei/Software/tvm/src/relay/op/tensor/transform.cc:251
#3  0x00007ffea00a352b in tvm::runtime::detail::unpack_call_dispatcher<tvm::runtime::Array<tvm::te::Tensor, void>, 0, 3, tvm::runtime::Array<tvm::te::Tensor, void> (*)(tvm::Attrs const&, tvm::runtime::Array<tvm::te::Tensor, void> const&, tvm::Type const&)>::run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_> (rv=0x7fffffff6e30, 
    args_pack=..., 
    f=@0x555556c96390: 0x7ffea030a79f <tvm::relay::ConcatenateCompute(tvm::Attrs const&, tvm::runtime::Array<tvm::te::Tensor, void> const&, tvm::Type const&)>, optional_name=0x0) at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1397
#4  tvm::runtime::detail::unpack_call_dispatcher<tvm::runtime::Array<tvm::te::Tensor, void>, 1, 2, tvm::runtime::Array<tvm::te::Tensor, void> (*)(tvm::Attrs const&, tvm::runtime::Array<tvm::te::Tensor, void> const&, tvm::Type const&)>::run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_> (rv=0x7fffffff6e30, args_pack=..., 
---Type <return> to continue, or q <return> to quit---
    teCompute(tvm::Attrs const&, tvm::runtime::Array<tvm::te::Tensor, void> const&, tvm::Type const&)>, optional_name=0x0) at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1382
#5  tvm::runtime::detail::unpack_call_dispatcher<tvm::runtime::Array<tvm::te::Tensor, void>, 2, 1, tvm::runtime::Array<tvm::te::Tensor, void> (*)(tvm::Attrs const&, tvm::runtime::Array<tvm::te::Tensor, void> const&, tvm::Type const&)>::run<tvm::runtime::TVMMovableArgValueWithContext_> (rv=0x7fffffff6e30, args_pack=..., 
    f=@0x555556c96390: 0x7ffea030a79f <tvm::relay::ConcatenateCompute(tvm::Attrs const&, tvm::runtime::Array<tvm::te::Tensor, void> const&, tvm::Type const&)>, optional_name=0x0) at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1382
#6  tvm::runtime::detail::unpack_call_dispatcher<tvm::runtime::Array<tvm::te::Tensor, void>, 3, 0, tvm::runtime::Array<tvm::te::Tensor, void> (*)(tvm::Attrs const&, tvm::runtime::Array<tvm::te::Tensor, void> const&, tvm::Type const&)>::run<>(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const*, tvm::runtime::Array<tvm::te::Tensor, void> (* const&)(tvm::Attrs const&, tvm::runtime::Array<tvm::te::Tensor, void> const&, tvm::Type const&), tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*) (rv=0x7fffffff6e30, 
    args_pack=..., 
    f=@0x555556c96390: 0x7ffea030a79f <tvm::relay::ConcatenateCompute(tvm::Attrs const&, tvm::runtime::Array<tvm::te::Tensor, void> const&, tvm::Type const&)>, optional_name=0x0) at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1382
#7  tvm::runtime::detail::unpack_call<tvm::runtime::Array<tvm::te::Tensor, void>, 3, tvm::runtime::Array<tvm::te::Tensor, void> (*)(tvm::Attrs const&, tvm::runtime::Array<tvm::te::Tensor, void> const&, tvm::Type const&)> (rv=0x7fffffff6e30, args=..., 
    f=@0x555556c96390: 0x7ffea030a79f <tvm::relay::ConcatenateCompute(tvm::Attrs const&, tvm::runtime::Array<tvm::te::Tensor, void> const&, tvm::Type const&)>, optional_name=0x0) at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1421
#8  void tvm::runtime::TypedPackedFunc<tvm::runtime::Array<tvm::te::Tensor, void> (tvm::Attrs const&, tvm::runtime::Array<tvm::te::Tensor, void> const&, tvm::Type const&)>::AssignTypedLambda<tvm::runtime::Array<tvm::te::Tensor, void> (*)(tvm::Attrs const&, tvm::runtime::Array<tvm::te::Tensor, void> const&, tvm::Type const&)>(tvm::runtime::Array<tvm::te::Tensor, void> (*)(tvm::Attrs const&, tvm::runtime::Array<tvm::te::Tensor, void> const&, tvm::Type const&))::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}::operator()(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*) const (__closure=0x555556c96390, args=..., rv=0x7fffffff6e30)
    at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1492
#9  0x00007ffea00a44f9 in std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), void tvm::runtime::TypedPackedFunc<tvm::runtime::Array<tvm::te::Tensor, void> (tvm::Attrs const&, tvm::runtime::Array<tvm::te::Tensor, void> const&, tvm::Type const&)>::AssignTypedLambda<tvm::runtime::Array<tvm::te::Tensor, void> (*)(tvm::Attrs const&, tvm::runtime::Array<tvm::te::Tensor, void> const&, tvm::Type const&)>(tvm::runtime::Array<tvm::te::Tensor, void> (*)(tvm::Attrs const&, tvm::runtime::Array<tvm::te::Tensor, void> const&, tvm::Type const&))::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&) (__functor=..., __args#0=..., __args#1=@0x7fffffff6cb0: 0x7fffffff6e30) at /usr/include/c++/7/bits/std_function.h:316
#10 0x00007ffe9f570cc0 in std::function<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)>::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const (this=0x555556c96390, __args#0=..., __args#1=0x7fffffff6e30) at /usr/include/c++/7/bits/std_function.h:706
#11 0x00007ffea08a211a in tvm::runtime::PackedFunc::operator()<tvm::Attrs const&, tvm::runtime::Array<tvm::te::Tensor, void> const&, tvm::Type const&> (this=0x555556c96390) at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1369
#12 0x00007ffea089e223 in tvm::runtime::detail::typed_packed_call_dispatcher<tvm::runtime::Array<tvm::te::Tensor, void> >::run<tvm::Attrs const&, tvm::runtime::Array<tvm::te::Tensor, void> const&, tvm::Type const&> (pf=...)
    at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1444
#13 tvm::runtime::TypedPackedFunc<tvm::runtime::Array<tvm::te::Tensor, void> (tvm::Attrs const&, tvm::runtime::Array<tvm::te::Tensor, void> const&, tvm::Type const&)>::operator()(tvm::Attrs const&, tvm::runtime::Array<tvm::te::Tensor, void> const&, tvm::Type const&) const (
    args#2=..., args#1=..., args#0=..., this=0x555556c96390) at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1498
#14 tvm::relay::OpImplementation::Compute (this=0x7fffffff6e80, attrs=..., inputs=..., out_type=...)
    at /home2/xiachunwei/Software/tvm/src/relay/ir/op_strategy.cc:36
#15 0x00007ffea089ead9 in tvm::relay::<lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)>::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue *) const (__closure=0x55555610f030, args=..., rv=0x7fffffff7020)
    at /home2/xiachunwei/Software/tvm/src/relay/ir/op_strategy.cc:80
#16 0x00007ffea089f496 in std::_Function_handler<void(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::relay::<lambda(tvm::runtime::T---Type <return> to continue, or q <return> to quit---
VMArgs, tvm::runtime::TVMRetValue*)> >::_M_invoke(const std::_Any_data &, tvm::runtime::TVMArgs &&, tvm::runtime::TVMRetValue *&&) (
    __functor=..., __args#0=..., __args#1=@0x7fffffff6f40: 0x7fffffff7020) at /usr/include/c++/7/bits/std_function.h:316
#17 0x00007ffe9f570cc0 in std::function<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)>::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const (this=0x55555610f030, __args#0=..., __args#1=0x7fffffff7020) at /usr/include/c++/7/bits/std_function.h:706
#18 0x00007ffe9f6dcf62 in tvm::runtime::PackedFunc::CallPacked (this=0x55555610f030, args=..., rv=0x7fffffff7020)
    at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1151
#19 0x00007ffea09c9c45 in TVMFuncCall (func=0x55555610f030, args=0x7ffe279e04b0, arg_type_codes=0x7ffe278cbbc0, num_args=4, 
    ret_val=0x7ffe278cb780, ret_type_code=0x7ffe278cb670) at /home2/xiachunwei/Software/tvm/src/runtime/c_runtime_api.cc:474
#20 0x00007ffff6535ec0 in ffi_call_unix64 () from /home/xiachunwei/Software/pytf2.4/lib/python3.7/lib-dynload/../../libffi.so.6
#21 0x00007ffff653587d in ffi_call () from /home/xiachunwei/Software/pytf2.4/lib/python3.7/lib-dynload/../../libffi.so.6
#22 0x00007ffff674bf3e in _call_function_pointer (argcount=6, resmem=0x7fffffff7220, restype=<optimized out>, atypes=0x7fffffff71a0, 
    avalues=0x7fffffff71e0, pProc=0x7ffea09c9bb0 <TVMFuncCall(TVMFunctionHandle, TVMValue*, int*, int, TVMValue*, int*)>, flags=4353)
   from /home/xiachunwei/Software/pytf2.4/lib/python3.7/lib-dynload/_ctypes.cpython-37m-x86_64-linux-gnu.so
#23 _ctypes_callproc () at <artificial>:1184
#24 0x00007ffff674c974 in PyCFuncPtr_call ()
   from /home/xiachunwei/Software/pytf2.4/lib/python3.7/lib-dynload/_ctypes.cpython-37m-x86_64-linux-gnu.so
#25 0x00005555556ce46b in _PyObject_FastCallKeywords () at /tmp/build/80754af9/python_1546061345851/work/Objects/call.c:199
#26 0x0000555555728d26 in call_function (kwnames=0x0, oparg=<optimized out>, pp_stack=<synthetic pointer>)
    at /tmp/build/80754af9/python_1546061345851/work/Python/ceval.c:4619
#27 _PyEval_EvalFrameDefault () at /tmp/build/80754af9/python_1546061345851/work/Python/ceval.c:3093
#28 0x00005555556690d9 in _PyEval_EvalCodeWithName () at /tmp/build/80754af9/python_1546061345851/work/Python/ceval.c:3930
#29 0x000055555566a1b4 in _PyFunction_FastCallDict () at /tmp/build/80754af9/python_1546061345851/work/Objects/call.c:376
#30 0x0000555555681473 in _PyObject_Call_Prepend () at /tmp/build/80754af9/python_1546061345851/work/Objects/call.c:904
#31 0x00005555556c594a in slot_tp_call () at /tmp/build/80754af9/python_1546061345851/work/Objects/typeobject.c:6376
#32 0x00005555556ce46b in _PyObject_FastCallKeywords () at /tmp/build/80754af9/python_1546061345851/work/Objects/call.c:199
#33 0x0000555555728646 in call_function (kwnames=0x0, oparg=<optimized out>, pp_stack=<synthetic pointer>)
    at /tmp/build/80754af9/python_1546061345851/work/Python/ceval.c:4619
#34 _PyEval_EvalFrameDefault () at /tmp/build/80754af9/python_1546061345851/work/Python/ceval.c:3124
#35 0x00005555556ccecb in function_code_fastcall (globals=<optimized out>, nargs=4, args=<optimized out>, co=<optimized out>)
    at /tmp/build/80754af9/python_1546061345851/work/Objects/call.c:283
#36 _PyFunction_FastCallKeywords () at /tmp/build/80754af9/python_1546061345851/work/Objects/call.c:408
#37 0x0000555555728561 in call_function (kwnames=0x0, oparg=<optimized out>, pp_stack=<synthetic pointer>)
    at /tmp/build/80754af9/python_1546061345851/work/Python/ceval.c:4616
#38 _PyEval_EvalFrameDefault () at /tmp/build/80754af9/python_1546061345851/work/Python/ceval.c:3093
#39 0x00005555556690d9 in _PyEval_EvalCodeWithName () at /tmp/build/80754af9/python_1546061345851/work/Python/ceval.c:3930
#40 0x00005555556cd0f5 in _PyFunction_FastCallKeywords () at /tmp/build/80754af9/python_1546061345851/work/Objects/call.c:433
#41 0x00005555557241a6 in call_function (kwnames=0x0, oparg=<optimized out>, pp_stack=<synthetic pointer>)
    at /tmp/build/80754af9/python_1546061345851/work/Python/ceval.c:4616
#42 _PyEval_EvalFrameDefault () at /tmp/build/80754af9/python_1546061345851/work/Python/ceval.c:3124
#43 0x000055555566a0eb in function_code_fastcall (globals=<optimized out>, nargs=3, args=<optimized out>, co=<optimized out>)
    at /tmp/build/80754af9/python_1546061345851/work/Objects/call.c:283
#44 _PyFunction_FastCallDict () at /tmp/build/80754af9/python_1546061345851/work/Objects/call.c:322
#45 0x00005555557259ff in do_call_core (kwdict=0x0, callargs=0x7ffe97648ee8, func=0x7ffe995047b8)
    at /tmp/build/80754af9/python_1546061345851/work/Python/ceval.c:4645
#46 _PyEval_EvalFrameDefault () at /tmp/build/80754af9/python_1546061345851/work/Python/ceval.c:3191
#47 0x00005555556693ba in _PyEval_EvalCodeWithName () at /tmp/build/80754af9/python_1546061345851/work/Python/ceval.c:3930
---Type <return> to continue, or q <return> to quit---
#48 0x000055555566a1b4 in _PyFunction_FastCallDict () at /tmp/build/80754af9/python_1546061345851/work/Objects/call.c:376
#49 0x00007ffff674ba11 in _CallPythonObject (pArgs=<optimized out>, flags=4353, converters=0x7ffea54488e0, callable=0x7ffe99504840, 
    setfunc=0x7ffff6745ea0 <i_set>, restype=0x7ffff685db88, mem=0x7fffffff8330)
   from /home/xiachunwei/Software/pytf2.4/lib/python3.7/lib-dynload/_ctypes.cpython-37m-x86_64-linux-gnu.so
#50 closure_fcn () at <artificial>:292
#51 0x00007ffff6535c60 in ffi_closure_unix64_inner () from /home/xiachunwei/Software/pytf2.4/lib/python3.7/lib-dynload/../../libffi.so.6
#52 0x00007ffff6536028 in ffi_closure_unix64 () from /home/xiachunwei/Software/pytf2.4/lib/python3.7/lib-dynload/../../libffi.so.6
#53 0x00007ffea09ca205 in <lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)>::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue *) const (__closure=0x5555560b9290, args=..., rv=0x7fffffff85d0) at /home2/xiachunwei/Software/tvm/src/runtime/c_runtime_api.cc:523
#54 0x00007ffea09cb630 in std::_Function_handler<void(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), TVMFuncCreateFromCFunc(TVMPackedCFunc, void*, TVMPackedCFuncFinalizer, void**)::<lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)> >::_M_invoke(const std::_Any_data &, tvm::runtime::TVMArgs &&, tvm::runtime::TVMRetValue *&&) (__functor=..., __args#0=..., __args#1=@0x7fffffff8440: 0x7fffffff85d0)
    at /usr/include/c++/7/bits/std_function.h:316
#55 0x00007ffe9f570cc0 in std::function<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)>::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const (this=0x5555560a8ba0, __args#0=..., __args#1=0x7fffffff85d0) at /usr/include/c++/7/bits/std_function.h:706
#56 0x00007ffea07c72fe in tvm::runtime::PackedFunc::operator()<tvm::relay::Call, tvm::runtime::Array<tvm::te::Tensor, void>&, tvm::Target&>
    (this=0x5555560a8ba0) at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1369
#57 0x00007ffea07c0b5c in tvm::relay::tec::ScheduleBuilder::VisitExpr_ (this=0x7fffffff8b80, call_node=0x555556cbfc40)
    at /home2/xiachunwei/Software/tvm/src/relay/backend/te_compiler_cache.cc:260
#58 0x00007ffea07c8a3d in tvm::relay::ExprFunctor<tvm::runtime::Array<tvm::te::Tensor, void> (tvm::RelayExpr const&)>::InitVTable()::{lambda(tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::runtime::Array<tvm::te::Tensor, void> (tvm::RelayExpr const&)>*)#6}::operator()(tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::runtime::Array<tvm::te::Tensor, void> (tvm::RelayExpr const&)>*) const (
    __closure=0x0, n=..., self=0x7fffffff8b80) at /home2/xiachunwei/Software/tvm/include/tvm/relay/expr_functor.h:128
#59 0x00007ffea07c8a98 in tvm::relay::ExprFunctor<tvm::runtime::Array<tvm::te::Tensor, void> (tvm::RelayExpr const&)>::InitVTable()::{lambda(tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::runtime::Array<tvm::te::Tensor, void> (tvm::RelayExpr const&)>*)#6}::_FUN(tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::runtime::Array<tvm::te::Tensor, void> (tvm::RelayExpr const&)>*) ()
    at /home2/xiachunwei/Software/tvm/include/tvm/relay/expr_functor.h:128
#60 0x00007ffea07c962e in tvm::NodeFunctor<tvm::runtime::Array<tvm::te::Tensor, void> (tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::runtime::Array<tvm::te::Tensor, void> (tvm::RelayExpr const&)>*)>::operator()(tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::runtime::Array<tvm::te::Tensor, void> (tvm::RelayExpr const&)>*) const (
    this=0x7ffea50ede50 <tvm::relay::ExprFunctor<tvm::runtime::Array<tvm::te::Tensor, void> (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)::vtable>, n=..., args#0=0x7fffffff8b80) at /home2/xiachunwei/Software/tvm/include/tvm/node/functor.h:97
#61 0x00007ffea07c75a0 in tvm::relay::ExprFunctor<tvm::runtime::Array<tvm::te::Tensor, void> (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&) (this=0x7fffffff8b80, n=...) at /home2/xiachunwei/Software/tvm/include/tvm/relay/expr_functor.h:95
#62 0x00007ffea07c6b8e in tvm::relay::backend::MemoizedExprTranslator<tvm::runtime::Array<tvm::te::Tensor, void> >::VisitExpr (
    this=0x7fffffff8b80, n=...) at /home2/xiachunwei/Software/tvm/src/relay/backend/utils.h:244
#63 0x00007ffea07bed4a in tvm::relay::tec::ScheduleBuilder::Create(tvm::relay::Function const&, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > (std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)>) (
    this=0x7fffffff8b80, prim_func=..., renamer=...) at /home2/xiachunwei/Software/tvm/src/relay/backend/te_compiler_cache.cc:134
#64 0x00007ffea07bc812 in tvm::relay::tec::PrimFuncFor(tvm::relay::Function const&, tvm::Target const&, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > (std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)>)
    (source_func=..., target=..., renamer=...) at /home2/xiachunwei/Software/tvm/src/relay/backend/te_compiler_cache.cc:354
#65 0x00007ffea07a453c in tvm::relay::tec::TECompilerImpl::LowerInternal(tvm::relay::tec::CCacheKey const&, std::function<tvm::runtime::String (tvm::runtime::String)>) (this=0x55555614d960, key=..., mangle_fn=...)
    at /home2/xiachunwei/Software/tvm/src/relay/backend/te_compiler.cc:236
#66 0x00007ffea07a1a91 in tvm::relay::tec::TECompilerImpl::Lower(tvm::relay::tec::CCacheKey const&, std::function<tvm::runtime::String (tvm::---Type <return> to continue, or q <return> to quit---
runtime::String)>) (this=0x55555614d960, key=..., mangle_fn=...) at /home2/xiachunwei/Software/tvm/src/relay/backend/te_compiler.cc:65
#67 0x00007ffea07a1c56 in tvm::relay::tec::TECompilerImpl::Lower (this=0x55555614d960, key=..., mod_name=...)
    at /home2/xiachunwei/Software/tvm/src/relay/backend/te_compiler.cc:71
#68 0x00007ffea07a6889 in tvm::relay::tec::LowerTensorExprMutator::LowerFunction (this=0x7fffffff9a80, func=..., target=...)
    at /home2/xiachunwei/Software/tvm/src/relay/backend/te_compiler.cc:484
#69 0x00007ffea07a8d20 in tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_ (this=0x7fffffff9a80, call_node=0x555556cd18b0)
    at /home2/xiachunwei/Software/tvm/src/relay/backend/te_compiler.cc:613
#70 0x00007ffea055a987 in tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_ (this=0x7fffffff9a80, call_node=0x555556cd18b0)
    at /home2/xiachunwei/Software/tvm/src/relay/transforms/device_aware_visitors.cc:267
#71 0x00007ffe9f72fadd in tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::InitVTable()::{lambda(tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>*)#6}::operator()(tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>*) const (__closure=0x0, n=..., self=0x7fffffff9a80)
    at /home2/xiachunwei/Software/tvm/include/tvm/relay/expr_functor.h:128
#72 0x00007ffe9f72fb38 in tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::InitVTable()::{lambda(tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>*)#6}::_FUN(tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>*) () at /home2/xiachunwei/Software/tvm/include/tvm/relay/expr_functor.h:128
#73 0x00007ffe9f7306ce in tvm::NodeFunctor<tvm::RelayExpr (tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>*)>::operator()(tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>*) const (
    this=0x7ffea50e3e00 <tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)::vtable>, n=..., 
    args#0=0x7fffffff9a80) at /home2/xiachunwei/Software/tvm/include/tvm/node/functor.h:97
#74 0x00007ffe9f72ea44 in tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&) (
    this=0x7fffffff9a80, n=...) at /home2/xiachunwei/Software/tvm/include/tvm/relay/expr_functor.h:95
#75 0x00007ffea08783b4 in tvm::relay::ExprMutator::VisitExpr (this=0x7fffffff9a80, expr=...)
    at /home2/xiachunwei/Software/tvm/src/relay/ir/expr_functor.cc:156
#76 0x00007ffea04c8428 in tvm::relay::ExprMutator::Mutate (this=0x7fffffff9a80, expr=...)
    at /home2/xiachunwei/Software/tvm/include/tvm/relay/expr_functor.h:190
#77 0x00007ffea0878bb4 in tvm::relay::ExprMutator::VisitExpr_ (this=0x7fffffff9a80, op=0x555556cb3e90)
    at /home2/xiachunwei/Software/tvm/src/relay/ir/expr_functor.cc:214
#78 0x00007ffea055aa2c in tvm::relay::transform::DeviceAwareExprMutator::DeviceAwareVisitExpr_ (this=0x7fffffff9a80, 
    function_node=0x555556cb3e90) at /home2/xiachunwei/Software/tvm/src/relay/transforms/device_aware_visitors.cc:272
#79 0x00007ffea055a26a in tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_ (this=0x7fffffff9a80, function_node=0x555556cb3e90)
    at /home2/xiachunwei/Software/tvm/src/relay/transforms/device_aware_visitors.cc:217
#80 0x00007ffe9f72fa01 in tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::InitVTable()::{lambda(tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>*)#5}::operator()(tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>*) const (__closure=0x0, n=..., self=0x7fffffff9a80)
    at /home2/xiachunwei/Software/tvm/include/tvm/relay/expr_functor.h:127
#81 0x00007ffe9f72fa5c in tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::InitVTable()::{lambda(tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>*)#5}::_FUN(tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>*) () at /home2/xiachunwei/Software/tvm/include/tvm/relay/expr_functor.h:127
#82 0x00007ffe9f7306ce in tvm::NodeFunctor<tvm::RelayExpr (tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>*)>::operator()(tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>*) const (
    this=0x7ffea50e3e00 <tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)::vtable>, n=..., 
    args#0=0x7fffffff9a80) at /home2/xiachunwei/Software/tvm/include/tvm/node/functor.h:97
#83 0x00007ffe9f72ea44 in tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&) (
    this=0x7fffffff9a80, n=...) at /home2/xiachunwei/Software/tvm/include/tvm/relay/expr_functor.h:95
#84 0x00007ffea08783b4 in tvm::relay::ExprMutator::VisitExpr (this=0x7fffffff9a80, expr=...)
---Type <return> to continue, or q <return> to quit---
    at /home2/xiachunwei/Software/tvm/src/relay/ir/expr_functor.cc:156
#85 0x00007ffea04c8428 in tvm::relay::ExprMutator::Mutate (this=0x7fffffff9a80, expr=...)
    at /home2/xiachunwei/Software/tvm/include/tvm/relay/expr_functor.h:190
#86 0x00007ffea0794744 in tvm::relay::tec::<lambda(tvm::relay::Function, tvm::IRModule, tvm::relay::transform::PassContext)>::operator()(tvm::relay::Function, tvm::IRModule, tvm::relay::transform::PassContext) const (__closure=0x555556cb7a90, func=..., module=..., ctx=...)
    at /home2/xiachunwei/Software/tvm/src/relay/backend/te_compiler.cc:678
#87 0x00007ffea079ade0 in tvm::runtime::detail::unpack_call_dispatcher<tvm::relay::Function, 0, 3, tvm::relay::tec::LowerTensorExpr(tvm::relay::tec::TargetMap, const tvm::runtime::String&, tvm::relay::tec::TECompiler, std::function<void(tvm::relay::Function)>)::<lambda(tvm::relay::Function, tvm::IRModule, tvm::relay::transform::PassContext)> >::run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_> (rv=0x7fffffffa040, args_pack=..., f=..., optional_name=0x0)
    at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1397
#88 tvm::runtime::detail::unpack_call_dispatcher<tvm::relay::Function, 1, 2, tvm::relay::tec::LowerTensorExpr(tvm::relay::tec::TargetMap, const tvm::runtime::String&, tvm::relay::tec::TECompiler, std::function<void(tvm::relay::Function)>)::<lambda(tvm::relay::Function, tvm::IRModule, tvm::relay::transform::PassContext)> >::run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_> (
    rv=0x7fffffffa040, args_pack=..., f=..., optional_name=0x0) at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1382
#89 tvm::runtime::detail::unpack_call_dispatcher<tvm::relay::Function, 2, 1, tvm::relay::tec::LowerTensorExpr(tvm::relay::tec::TargetMap, const tvm::runtime::String&, tvm::relay::tec::TECompiler, std::function<void(tvm::relay::Function)>)::<lambda(tvm::relay::Function, tvm::IRModule, tvm::relay::transform::PassContext)> >::run<tvm::runtime::TVMMovableArgValueWithContext_> (rv=0x7fffffffa040, args_pack=..., f=..., 
    optional_name=0x0) at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1382
#90 tvm::runtime::detail::unpack_call_dispatcher<tvm::relay::Function, 3, 0, tvm::relay::tec::LowerTensorExpr(tvm::relay::tec::TargetMap, const tvm::runtime::String&, tvm::relay::tec::TECompiler, std::function<void(tvm::relay::Function)>)::<lambda(tvm::relay::Function, tvm::IRModule, tvm::relay::transform::PassContext)> >::run<> (rv=0x7fffffffa040, args_pack=..., f=..., optional_name=0x0)
    at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1382
#91 tvm::runtime::detail::unpack_call<tvm::relay::Function, 3, tvm::relay::tec::LowerTensorExpr(tvm::relay::tec::TargetMap, const tvm::runtime::String&, tvm::relay::tec::TECompiler, std::function<void(tvm::relay::Function)>)::<lambda(tvm::relay::Function, tvm::IRModule, tvm::relay::transform::PassContext)> > (rv=0x7fffffffa040, args=..., f=..., optional_name=0x0)
    at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1421
#92 tvm::runtime::TypedPackedFunc<tvm::relay::Function(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)>::<lambda(const tvm::runtime::TVMArgs&, tvm::runtime::TVMRetValue*)>::operator()(const tvm::runtime::TVMArgs &, tvm::runtime::TVMRetValue *) const (
    __closure=0x555556cb7a90, args=..., rv=0x7fffffffa040) at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1492
#93 0x00007ffea079edbb in std::_Function_handler<void(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<R(Args ...)>::AssignTypedLambda(FType) [with FLambda = tvm::relay::tec::LowerTensorExpr(tvm::relay::tec::TargetMap, const tvm::runtime::String&, tvm::relay::tec::TECompiler, std::function<void(tvm::relay::Function)>)::<lambda(tvm::relay::Function, tvm::IRModule, tvm::relay::transform::PassContext)>; R = tvm::relay::Function; Args = {tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext}]::<lambda(const tvm::runtime::TVMArgs&, tvm::runtime::TVMRetValue*)> >::_M_invoke(const std::_Any_data &, tvm::runtime::TVMArgs &&, tvm::runtime::TVMRetValue *&&) (
    __functor=..., __args#0=..., __args#1=@0x7fffffff9e20: 0x7fffffffa040) at /usr/include/c++/7/bits/std_function.h:316
#94 0x00007ffe9f570cc0 in std::function<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)>::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const (this=0x555556cc5b00, __args#0=..., __args#1=0x7fffffffa040) at /usr/include/c++/7/bits/std_function.h:706
#95 0x00007ffea08a8ae0 in tvm::runtime::PackedFunc::operator()<tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext> (
    this=0x555556cc5b00) at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1369
#96 0x00007ffea08a60e8 in tvm::runtime::detail::typed_packed_call_dispatcher<tvm::relay::Function>::run<tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext> (pf=...) at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1444
#97 tvm::runtime::TypedPackedFunc<tvm::relay::Function (tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)>::operator()(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext) const (args#2=..., args#1=..., args#0=..., this=0x555556cc5b00)
    at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1498
#98 tvm::relay::transform::FunctionPassNode::operator() (this=0x555556cc5ae0, mod=..., pass_ctx=...)
---Type <return> to continue, or q <return> to quit---
    at /home2/xiachunwei/Software/tvm/src/relay/ir/transform.cc:144
#99 0x00007ffe9f75b1fc in tvm::transform::Pass::operator() (this=0x7fffffffa6d0, mod=..., pass_ctx=...)
    at /home2/xiachunwei/Software/tvm/src/ir/transform.cc:267
#100 0x00007ffe9f75aeae in tvm::transform::Pass::operator() (this=0x7fffffffa6d0, mod=...)
    at /home2/xiachunwei/Software/tvm/src/ir/transform.cc:255
#101 0x00007ffea079891b in tvm::relay::tec::LowerTE(tvm::IRModule const&, std::unordered_map<DLDeviceType, tvm::Target, tvm::relay::backend::EnumClassHash, std::equal_to<DLDeviceType>, std::allocator<std::pair<DLDeviceType const, tvm::Target> > >, tvm::runtime::String const&, std::function<void (tvm::relay::Function)>) (module=..., targets=std::unordered_map with 1 element = {...}, module_name=..., process_fn=...)
    at /home2/xiachunwei/Software/tvm/src/relay/backend/te_compiler.cc:903
#102 0x00007ffea07994be in tvm::relay::tec::<lambda(tvm::IRModule, tvm::relay::transform::PassContext)>::operator()(tvm::IRModule, tvm::relay::transform::PassContext) const (__closure=0x555556cb5100, module=..., ctx=...)
    at /home2/xiachunwei/Software/tvm/src/relay/backend/te_compiler.cc:953
#103 0x00007ffea079b598 in tvm::runtime::detail::unpack_call_dispatcher<tvm::IRModule, 0, 2, tvm::relay::tec::LowerTEPass(tvm::relay::tec::TargetMap, const tvm::runtime::String&, std::function<void(tvm::relay::Function)>)::<lambda(tvm::IRModule, tvm::relay::transform::PassContext)> >::run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_> (rv=0x7fffffffacc0, args_pack=..., 
    f=..., optional_name=0x0) at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1397
#104 tvm::runtime::detail::unpack_call_dispatcher<tvm::IRModule, 1, 1, tvm::relay::tec::LowerTEPass(tvm::relay::tec::TargetMap, const tvm::runtime::String&, std::function<void(tvm::relay::Function)>)::<lambda(tvm::IRModule, tvm::relay::transform::PassContext)> >::run<tvm::runtime::TVMMovableArgValueWithContext_> (rv=0x7fffffffacc0, args_pack=..., f=..., optional_name=0x0)
    at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1382
#105 tvm::runtime::detail::unpack_call_dispatcher<tvm::IRModule, 2, 0, tvm::relay::tec::LowerTEPass(tvm::relay::tec::TargetMap, const tvm::runtime::String&, std::function<void(tvm::relay::Function)>)::<lambda(tvm::IRModule, tvm::relay::transform::PassContext)> >::run<> (
    rv=0x7fffffffacc0, args_pack=..., f=..., optional_name=0x0) at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1382
#106 tvm::runtime::detail::unpack_call<tvm::IRModule, 2, tvm::relay::tec::LowerTEPass(tvm::relay::tec::TargetMap, const tvm::runtime::String&, std::function<void(tvm::relay::Function)>)::<lambda(tvm::IRModule, tvm::relay::transform::PassContext)> > (rv=0x7fffffffacc0, args=..., 
    f=..., optional_name=0x0) at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1421
#107 tvm::runtime::TypedPackedFunc<tvm::IRModule(tvm::IRModule, tvm::transform::PassContext)>::<lambda(const tvm::runtime::TVMArgs&, tvm::runtime::TVMRetValue*)>::operator()(const tvm::runtime::TVMArgs &, tvm::runtime::TVMRetValue *) const (__closure=0x555556cb5100, args=..., 
    rv=0x7fffffffacc0) at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1492
#108 0x00007ffea079ef56 in std::_Function_handler<void(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<R(Args ...)>::AssignTypedLambda(FType) [with FLambda = tvm::relay::tec::LowerTEPass(tvm::relay::tec::TargetMap, const tvm::runtime::String&, std::function<void(tvm::relay::Function)>)::<lambda(tvm::IRModule, tvm::relay::transform::PassContext)>; R = tvm::IRModule; Args = {tvm::IRModule, tvm::transform::PassContext}]::<lambda(const tvm::runtime::TVMArgs&, tvm::runtime::TVMRetValue*)> >::_M_invoke(const std::_Any_data &, tvm::runtime::TVMArgs &&, tvm::runtime::TVMRetValue *&&) (__functor=..., __args#0=..., __args#1=@0x7fffffffab50: 0x7fffffffacc0)
    at /usr/include/c++/7/bits/std_function.h:316
#109 0x00007ffe9f570cc0 in std::function<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)>::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const (this=0x555556caf8f0, __args#0=..., __args#1=0x7fffffffacc0) at /usr/include/c++/7/bits/std_function.h:706
#110 0x00007ffe9f76bc72 in tvm::runtime::PackedFunc::operator()<tvm::IRModule, tvm::transform::PassContext> (this=0x555556caf8f0)
    at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1369
#111 0x00007ffe9f75bd36 in tvm::runtime::detail::typed_packed_call_dispatcher<tvm::IRModule>::run<tvm::IRModule, tvm::transform::PassContext> (pf=...) at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1444
#112 tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::operator()(tvm::IRModule, tvm::transform::PassContext) const (args#1=..., args#0=..., this=0x555556caf8f0) at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1498
#113 tvm::transform::ModulePassNode::operator() (this=0x555556caf8d0, mod=..., pass_ctx=...)
    at /home2/xiachunwei/Software/tvm/src/ir/transform.cc:416
#114 0x00007ffe9f75b1fc in tvm::transform::Pass::operator() (this=0x7fffffffb2b8, mod=..., pass_ctx=...)
---Type <return> to continue, or q <return> to quit---
    at /home2/xiachunwei/Software/tvm/src/ir/transform.cc:267
#115 0x00007ffe9f75d051 in tvm::transform::SequentialNode::operator() (this=0x555556cc35c0, mod=..., pass_ctx=...)
    at /home2/xiachunwei/Software/tvm/src/ir/transform.cc:487
#116 0x00007ffe9f75b1fc in tvm::transform::Pass::operator() (this=0x7fffffffb840, mod=..., pass_ctx=...)
    at /home2/xiachunwei/Software/tvm/src/ir/transform.cc:267
#117 0x00007ffe9f75aeae in tvm::transform::Pass::operator() (this=0x7fffffffb840, mod=...)
    at /home2/xiachunwei/Software/tvm/src/ir/transform.cc:255
#118 0x00007ffea07341be in tvm::relay::backend::GraphExecutorCodegen::Codegen (this=0x555556cb6820, func=..., mod_name=...)
    at /home2/xiachunwei/Software/tvm/src/relay/backend/graph_executor_codegen.cc:238
#119 0x00007ffea0739edf in tvm::relay::backend::GraphExecutorCodegenModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#2}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const (__closure=0x555556a84540, args=..., rv=0x7fffffffbed0)
    at /home2/xiachunwei/Software/tvm/src/relay/backend/graph_executor_codegen.cc:636
#120 0x00007ffea07452f5 in std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::relay::backend::GraphExecutorCodegenModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#2}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&) (__functor=..., __args#0=..., __args#1=@0x7fffffffbdd0: 0x7fffffffbed0)
    at /usr/include/c++/7/bits/std_function.h:316
#121 0x00007ffe9f570cc0 in std::function<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)>::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const (this=0x7fffffffbee0, __args#0=..., __args#1=0x7fffffffbed0) at /usr/include/c++/7/bits/std_function.h:706
#122 0x00007ffea071fc1c in tvm::runtime::PackedFunc::operator()<tvm::relay::Function, tvm::runtime::String> (this=0x7fffffffbee0)
    at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1369
#123 0x00007ffea071dc0e in tvm::relay::backend::ExecutorCodegen::CallFunc<tvm::relay::Function, tvm::runtime::String> (this=0x555556cb5c10, 
    name="codegen") at /home2/xiachunwei/Software/tvm/src/relay/backend/build_module.cc:112
#124 0x00007ffea07168ff in tvm::relay::backend::ExecutorCodegen::Codegen (this=0x555556cb5c10, func=..., mod_name=...)
    at /home2/xiachunwei/Software/tvm/src/relay/backend/build_module.cc:61
#125 0x00007ffea071c09c in tvm::relay::backend::RelayBuildModule::BuildRelay (this=0x55555621b830, relay_module=..., 
    params=std::unordered_map with 0 elements, mod_name=...) at /home2/xiachunwei/Software/tvm/src/relay/backend/build_module.cc:459
#126 0x00007ffea0719c75 in tvm::relay::backend::RelayBuildModule::Build (this=0x55555621b830, mod=..., targets=..., target_host=..., 
    executor=..., mod_name=...) at /home2/xiachunwei/Software/tvm/src/relay/backend/build_module.cc:297
#127 0x00007ffea0717cb5 in tvm::relay::backend::RelayBuildModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#3}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const (__closure=0x555556acb1d0, args=..., rv=0x7fffffffc610)
    at /home2/xiachunwei/Software/tvm/src/relay/backend/build_module.cc:181
#128 0x00007ffea07206c2 in std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::relay::backend::RelayBuildModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#3}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&) (__functor=..., __args#0=..., __args#1=@0x7fffffffc530: 0x7fffffffc610)
    at /usr/include/c++/7/bits/std_function.h:316
#129 0x00007ffe9f570cc0 in std::function<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)>::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const (this=0x555556c48bd0, __args#0=..., __args#1=0x7fffffffc610) at /usr/include/c++/7/bits/std_function.h:706
#130 0x00007ffe9f6dcf62 in tvm::runtime::PackedFunc::CallPacked (this=0x555556c48bd0, args=..., rv=0x7fffffffc610)
    at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1151
#131 0x00007ffea09c9c45 in TVMFuncCall (func=0x555556c48bd0, args=0x7ffe279e2698, arg_type_codes=0x7ffe27a3ba08, num_args=5, 
    ret_val=0x7ffe2793db38, ret_type_code=0x7ffe2793df78) at /home2/xiachunwei/Software/tvm/src/runtime/c_runtime_api.cc:474
#132 0x00007ffff6535ec0 in ffi_call_unix64 () from /home/xiachunwei/Software/pytf2.4/lib/python3.7/lib-dynload/../../libffi.so.6
---Type <return> to continue, or q <return> to quit---
#133 0x00007ffff653587d in ffi_call () from /home/xiachunwei/Software/pytf2.4/lib/python3.7/lib-dynload/../../libffi.so.6
#134 0x00007ffff674bf3e in _call_function_pointer (argcount=6, resmem=0x7fffffffc810, restype=<optimized out>, atypes=0x7fffffffc790, 
    avalues=0x7fffffffc7d0, pProc=0x7ffea09c9bb0 <TVMFuncCall(TVMFunctionHandle, TVMValue*, int*, int, TVMValue*, int*)>, flags=4353)
   from /home/xiachunwei/Software/pytf2.4/lib/python3.7/lib-dynload/_ctypes.cpython-37m-x86_64-linux-gnu.so
#135 _ctypes_callproc () at <artificial>:1184
#136 0x00007ffff674c974 in PyCFuncPtr_call ()
   from /home/xiachunwei/Software/pytf2.4/lib/python3.7/lib-dynload/_ctypes.cpython-37m-x86_64-linux-gnu.so
#137 0x00005555556ce46b in _PyObject_FastCallKeywords () at /tmp/build/80754af9/python_1546061345851/work/Objects/call.c:199
#138 0x0000555555728d26 in call_function (kwnames=0x0, oparg=<optimized out>, pp_stack=<synthetic pointer>)
    at /tmp/build/80754af9/python_1546061345851/work/Python/ceval.c:4619
#139 _PyEval_EvalFrameDefault () at /tmp/build/80754af9/python_1546061345851/work/Python/ceval.c:3093
#140 0x00005555556690d9 in _PyEval_EvalCodeWithName () at /tmp/build/80754af9/python_1546061345851/work/Python/ceval.c:3930
#141 0x000055555566a1b4 in _PyFunction_FastCallDict () at /tmp/build/80754af9/python_1546061345851/work/Objects/call.c:376
#142 0x00005555556814ee in _PyObject_Call_Prepend () at /tmp/build/80754af9/python_1546061345851/work/Objects/call.c:904
#143 0x00005555556c594a in slot_tp_call () at /tmp/build/80754af9/python_1546061345851/work/Objects/typeobject.c:6376
#144 0x00005555556ce46b in _PyObject_FastCallKeywords () at /tmp/build/80754af9/python_1546061345851/work/Objects/call.c:199
#145 0x0000555555728d26 in call_function (kwnames=0x0, oparg=<optimized out>, pp_stack=<synthetic pointer>)
    at /tmp/build/80754af9/python_1546061345851/work/Python/ceval.c:4619
#146 _PyEval_EvalFrameDefault () at /tmp/build/80754af9/python_1546061345851/work/Python/ceval.c:3093
#147 0x00005555556690d9 in _PyEval_EvalCodeWithName () at /tmp/build/80754af9/python_1546061345851/work/Python/ceval.c:3930
#148 0x00005555556cd157 in _PyFunction_FastCallKeywords () at /tmp/build/80754af9/python_1546061345851/work/Objects/call.c:433
#149 0x0000555555724fa1 in call_function (kwnames=0x7ffe994da830, oparg=<optimized out>, pp_stack=<synthetic pointer>)
    at /tmp/build/80754af9/python_1546061345851/work/Python/ceval.c:4616
#150 _PyEval_EvalFrameDefault () at /tmp/build/80754af9/python_1546061345851/work/Python/ceval.c:3139
#151 0x00005555556690d9 in _PyEval_EvalCodeWithName () at /tmp/build/80754af9/python_1546061345851/work/Python/ceval.c:3930
#152 0x00005555556cd157 in _PyFunction_FastCallKeywords () at /tmp/build/80754af9/python_1546061345851/work/Objects/call.c:433
#153 0x0000555555724fa1 in call_function (kwnames=0x7ffff690f7b8, oparg=<optimized out>, pp_stack=<synthetic pointer>)
    at /tmp/build/80754af9/python_1546061345851/work/Python/ceval.c:4616
#154 _PyEval_EvalFrameDefault () at /tmp/build/80754af9/python_1546061345851/work/Python/ceval.c:3139
#155 0x0000555555669899 in _PyEval_EvalCodeWithName () at /tmp/build/80754af9/python_1546061345851/work/Python/ceval.c:3930
#156 0x00005555556cd157 in _PyFunction_FastCallKeywords () at /tmp/build/80754af9/python_1546061345851/work/Objects/call.c:433
#157 0x00005555557241a6 in call_function (kwnames=0x0, oparg=<optimized out>, pp_stack=<synthetic pointer>)
    at /tmp/build/80754af9/python_1546061345851/work/Python/ceval.c:4616
#158 _PyEval_EvalFrameDefault () at /tmp/build/80754af9/python_1546061345851/work/Python/ceval.c:3124
#159 0x00005555556690d9 in _PyEval_EvalCodeWithName () at /tmp/build/80754af9/python_1546061345851/work/Python/ceval.c:3930
#160 0x0000555555669fa4 in PyEval_EvalCodeEx () at /tmp/build/80754af9/python_1546061345851/work/Python/ceval.c:3959
#161 0x0000555555669fcc in PyEval_EvalCode (co=<optimized out>, globals=<optimized out>, locals=<optimized out>)
    at /tmp/build/80754af9/python_1546061345851/work/Python/ceval.c:524
#162 0x0000555555783664 in run_mod () at /tmp/build/80754af9/python_1546061345851/work/Python/pythonrun.c:1035
#163 0x000055555578d881 in PyRun_FileExFlags () at /tmp/build/80754af9/python_1546061345851/work/Python/pythonrun.c:988
#164 0x000055555578da73 in PyRun_SimpleFileExFlags () at /tmp/build/80754af9/python_1546061345851/work/Python/pythonrun.c:429
#165 0x000055555578eb2f in pymain_run_file (p_cf=0x7fffffffd920, filename=0x5555558c68a0 L"relay_ir.py", fp=0x555555907b20)
    at /tmp/build/80754af9/python_1546061345851/work/Modules/main.c:427
#166 pymain_run_filename (cf=0x7fffffffd920, pymain=0x7fffffffda30) at /tmp/build/80754af9/python_1546061345851/work/Modules/main.c:1627
#167 pymain_run_python (pymain=0x7fffffffda30) at /tmp/build/80754af9/python_1546061345851/work/Modules/main.c:2876
#168 pymain_main () at /tmp/build/80754af9/python_1546061345851/work/Modules/main.c:3037
---Type <return> to continue, or q <return> to quit---
#169 0x000055555578ec4c in _Py_UnixMain () at /tmp/build/80754af9/python_1546061345851/work/Modules/main.c:3072
#170 0x00007ffff77e6b97 in __libc_start_main (main=0x555555649540 <main>, argc=2, argv=0x7fffffffdb88, init=<optimized out>, 
    fini=<optimized out>, rtld_fini=<optimized out>, stack_end=0x7fffffffdb78) at ../csu/libc-start.c:310
#171 0x0000555555733982 in _start () at ../sysdeps/x86_64/elf/start.S:103
```

### python层面build的stack

```Python
Traceback (most recent call last):
  File "test_conv2d_relu_2branches.py", line 58, in <module>
    test_conv2d_relu_2way()
  File "test_conv2d_relu_2branches.py", line 42, in test_conv2d_relu_2way
    lib = relay.build(mod, target, params=params)
  File "/home2/xiachunwei/Software/tvm/python/tvm/relay/build_module.py", line 363, in build
    mod=ir_mod, target=target, params=params, executor=executor, mod_name=mod_name
  File "/home2/xiachunwei/Software/tvm/python/tvm/relay/build_module.py", line 172, in build
    self._build(mod, target, target_host, executor, mod_name)
  File "/home2/xiachunwei/Software/tvm/python/tvm/_ffi/_ctypes/packed_func.py", line 237, in __call__
    raise get_last_ffi_error()
tvm._ffi.base.TVMError: Traceback (most recent call last):
  14: TVMFuncCall
        at /home2/xiachunwei/Software/tvm/src/runtime/c_runtime_api.cc:474
  13: tvm::runtime::PackedFunc::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1151
  12: std::function<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)>::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at /usr/include/c++/7/bits/std_function.h:706
  11: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::relay::backend::RelayBuildModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#3}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
        at /usr/include/c++/7/bits/std_function.h:316
  10: tvm::relay::backend::RelayBuildModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#3}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at /home2/xiachunwei/Software/tvm/src/relay/backend/build_module.cc:181
  9: tvm::relay::backend::RelayBuildModule::Build(tvm::IRModule, tvm::runtime::Map<tvm::Integer, tvm::Target, void, void> const&, tvm::Target const&, tvm::runtime::String, tvm::runtime::String)
        at /home2/xiachunwei/Software/tvm/src/relay/backend/build_module.cc:297
  8: tvm::relay::backend::RelayBuildModule::BuildRelay(tvm::IRModule, std::unordered_map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, tvm::runtime::NDArray, std::hash<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::equal_to<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, tvm::runtime::NDArray> > > const&, tvm::runtime::String)
        at /home2/xiachunwei/Software/tvm/src/relay/backend/build_module.cc:461
  7: tvm::relay::backend::ExecutorCodegen::Codegen(tvm::relay::Function const&, tvm::runtime::String)
        at /home2/xiachunwei/Software/tvm/src/relay/backend/build_module.cc:61
  6: void tvm::relay::backend::ExecutorCodegen::CallFunc<tvm::relay::Function, tvm::runtime::String>(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::relay::Function, tvm::runtime::String)
        at /home2/xiachunwei/Software/tvm/src/relay/backend/build_module.cc:112
  5: tvm::runtime::TVMRetValue tvm::runtime::PackedFunc::operator()<tvm::relay::Function, tvm::runtime::String>(tvm::relay::Function&&, tvm::runtime::String&&) const
        at /home2/xiachunwei/Software/tvm/include/tvm/runtime/packed_func.h:1369
  4: std::function<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)>::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at /usr/include/c++/7/bits/std_function.h:706
  3: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::relay::backend::GraphExecutorCodegenModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#2}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
        at /usr/include/c++/7/bits/std_function.h:316
  2: tvm::relay::backend::GraphExecutorCodegenModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#2}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at /home2/xiachunwei/Software/tvm/src/relay/backend/graph_executor_codegen.cc:638
  1: tvm::relay::backend::GraphExecutorCodegen::Codegen(tvm::relay::Function, tvm::runtime::String)
        at /home2/xiachunwei/Software/tvm/src/relay/backend/graph_executor_codegen.cc:284
  0: tvm::relay::GetPerVarTensorsFromIRModule(tvm::IRModule const&)
        at /home2/xiachunwei/Software/tvm/src/relay/backend/tir_attr_metadata_visitor.cc:49
  File "/home2/xiachunwei/Software/tvm/src/relay/backend/tir_attr_metadata_visitor.cc", line 49
TVMError: 
---------------------------------------------------------------
An error occurred during the execution of TVM.
For more information, please see: https://tvm.apache.org/docs/errors.html
---------------------------------------------------------------
  Check failed: (0) is false: 

```