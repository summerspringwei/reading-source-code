Traceback (most recent call last):
  File "test_conv2d_branches.py", line 205, in <module>
    run_hand_schedule()
  File "test_conv2d_branches.py", line 101, in run_hand_schedule
    mod = tvm.build(sch, [x1, x2, T_relu], target=tgt_gpu, name="hand_schedule")
  File "/home/xiachunwei/Software/clean_tvm/tvm/python/tvm/driver/build_module.py", line 219, in build
    input_mod = lower(inputs, args, name=name, binds=binds)
  File "/home/xiachunwei/Software/clean_tvm/tvm/python/tvm/driver/build_module.py", line 133, in lower
    return ffi.lower_schedule(inp, args, name, binds, simple_mode)
  File "/home/xiachunwei/Software/clean_tvm/tvm/python/tvm/_ffi/_ctypes/packed_func.py", line 237, in __call__
    raise get_last_ffi_error()
tvm._ffi.base.TVMError: Traceback (most recent call last):
  42: TVMFuncCall
        at /home/xiachunwei/Software/clean_tvm/tvm/src/runtime/c_runtime_api.cc:474
  41: tvm::runtime::PackedFunc::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at /home/xiachunwei/Software/clean_tvm/tvm/include/tvm/runtime/packed_func.h:1151
  40: std::function<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)>::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at /usr/include/c++/7/bits/std_function.h:706
  39: _M_invoke
        at /usr/include/c++/7/bits/std_function.h:316
  38: operator()
        at /home/xiachunwei/Software/clean_tvm/tvm/include/tvm/runtime/packed_func.h:1480
  37: unpack_call<tvm::IRModule, 5, tvm::<lambda(tvm::te::Schedule, const tvm::runtime::Array<tvm::runtime::ObjectRef>&, const tvm::runtime::String&, const tvm::runtime::Map<tvm::te::Tensor, tvm::tir::Buffer>&, bool)> >
        at /home/xiachunwei/Software/clean_tvm/tvm/include/tvm/runtime/packed_func.h:1421
  36: run<>
        at /home/xiachunwei/Software/clean_tvm/tvm/include/tvm/runtime/packed_func.h:1382
  35: run<tvm::runtime::TVMMovableArgValueWithContext_>
        at /home/xiachunwei/Software/clean_tvm/tvm/include/tvm/runtime/packed_func.h:1382
  34: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at /home/xiachunwei/Software/clean_tvm/tvm/include/tvm/runtime/packed_func.h:1382
  33: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at /home/xiachunwei/Software/clean_tvm/tvm/include/tvm/runtime/packed_func.h:1382
  32: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at /home/xiachunwei/Software/clean_tvm/tvm/include/tvm/runtime/packed_func.h:1382
  31: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at /home/xiachunwei/Software/clean_tvm/tvm/include/tvm/runtime/packed_func.h:1397
  30: operator()
        at /home/xiachunwei/Software/clean_tvm/tvm/src/driver/driver_api.cc:395
  29: tvm::LowerSchedule(tvm::te::Schedule, tvm::runtime::Array<tvm::runtime::ObjectRef, void> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer, std::hash<tvm::te::Tensor>, std::equal_to<tvm::te::Tensor>, std::allocator<std::pair<tvm::te::Tensor const, tvm::tir::Buffer> > > const&, bool)
        at /home/xiachunwei/Software/clean_tvm/tvm/src/driver/driver_api.cc:379
  28: tvm::ScheduleToModule(tvm::te::Schedule, tvm::runtime::Array<tvm::runtime::ObjectRef, void> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer, std::hash<tvm::te::Tensor>, std::equal_to<tvm::te::Tensor>, std::allocator<std::pair<tvm::te::Tensor const, tvm::tir::Buffer> > > const&)
        at /home/xiachunwei/Software/clean_tvm/tvm/src/driver/driver_api.cc:294
  27: tvm::te::Schedule::normalize()
        at /home/xiachunwei/Software/clean_tvm/tvm/src/te/schedule/schedule_dataflow_rewrite.cc:716
  26: tvm::te::InjectInline(tvm::te::ScheduleNode*, bool)
        at /home/xiachunwei/Software/clean_tvm/tvm/src/te/schedule/schedule_dataflow_rewrite.cc:606
  25: tvm::te::ComputeOp::ComputeOp(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, tvm::runtime::Map<tvm::runtime::String, tvm::runtime::ObjectRef, void, void>, tvm::runtime::Array<tvm::tir::IterVar, void>, tvm::runtime::Array<tvm::PrimExpr, void>)
        at /home/xiachunwei/Software/clean_tvm/tvm/src/te/operation/compute_op.cc:144
  24: VerifyComputeOp
        at /home/xiachunwei/Software/clean_tvm/tvm/src/te/operation/compute_op.cc:566
  23: Run
        at /home/xiachunwei/Software/clean_tvm/tvm/src/te/operation/compute_op.cc:536
  22: tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>::VisitExpr(tvm::PrimExpr const&)
        at /home/xiachunwei/Software/clean_tvm/tvm/include/tvm/tir/expr_functor.h:114
  21: tvm::NodeFunctor<void (tvm::runtime::ObjectRef const&, tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>*)>::operator()(tvm::runtime::ObjectRef const&, tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>*) const
        at /home/xiachunwei/Software/clean_tvm/tvm/include/tvm/node/functor.h:97
  20: tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>::InitVTable()::{lambda(tvm::runtime::ObjectRef const&, tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>*)#16}::_FUN(tvm::runtime::ObjectRef const&, tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>*)
        at /home/xiachunwei/Software/clean_tvm/tvm/include/tvm/tir/expr_functor.h:179
  19: tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>::InitVTable()::{lambda(tvm::runtime::ObjectRef const&, tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>*)#16}::operator()(tvm::runtime::ObjectRef const&, tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>*) const
        at /home/xiachunwei/Software/clean_tvm/tvm/include/tvm/tir/expr_functor.h:179
  18: tvm::tir::ExprVisitor::VisitExpr_(tvm::tir::MaxNode const*)
        at /home/xiachunwei/Software/clean_tvm/tvm/src/tir/ir/expr_functor.cc:73
  17: VisitExpr
        at /home/xiachunwei/Software/clean_tvm/tvm/src/te/operation/compute_op.cc:545
  16: tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>::VisitExpr(tvm::PrimExpr const&)
        at /home/xiachunwei/Software/clean_tvm/tvm/include/tvm/tir/expr_functor.h:114
  15: tvm::NodeFunctor<void (tvm::runtime::ObjectRef const&, tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>*)>::operator()(tvm::runtime::ObjectRef const&, tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>*) const
        at /home/xiachunwei/Software/clean_tvm/tvm/include/tvm/node/functor.h:97
  14: tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>::InitVTable()::{lambda(tvm::runtime::ObjectRef const&, tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>*)#10}::_FUN(tvm::runtime::ObjectRef const&, tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>*)
        at /home/xiachunwei/Software/clean_tvm/tvm/include/tvm/tir/expr_functor.h:173
  13: tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>::InitVTable()::{lambda(tvm::runtime::ObjectRef const&, tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>*)#10}::operator()(tvm::runtime::ObjectRef const&, tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>*) const
        at /home/xiachunwei/Software/clean_tvm/tvm/include/tvm/tir/expr_functor.h:173
  12: tvm::tir::ExprVisitor::VisitExpr_(tvm::tir::MulNode const*)
        at /home/xiachunwei/Software/clean_tvm/tvm/src/tir/ir/expr_functor.cc:67
  11: VisitExpr
        at /home/xiachunwei/Software/clean_tvm/tvm/src/te/operation/compute_op.cc:545
  10: tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>::VisitExpr(tvm::PrimExpr const&)
        at /home/xiachunwei/Software/clean_tvm/tvm/include/tvm/tir/expr_functor.h:114
  9: tvm::NodeFunctor<void (tvm::runtime::ObjectRef const&, tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>*)>::operator()(tvm::runtime::ObjectRef const&, tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>*) const
        at /home/xiachunwei/Software/clean_tvm/tvm/include/tvm/node/functor.h:97
  8: tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>::InitVTable()::{lambda(tvm::runtime::ObjectRef const&, tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>*)#10}::_FUN(tvm::runtime::ObjectRef const&, tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>*)
        at /home/xiachunwei/Software/clean_tvm/tvm/include/tvm/tir/expr_functor.h:173
  7: tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>::InitVTable()::{lambda(tvm::runtime::ObjectRef const&, tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>*)#10}::operator()(tvm::runtime::ObjectRef const&, tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>*) const
        at /home/xiachunwei/Software/clean_tvm/tvm/include/tvm/tir/expr_functor.h:173
  6: tvm::tir::ExprVisitor::VisitExpr_(tvm::tir::MulNode const*)
        at /home/xiachunwei/Software/clean_tvm/tvm/src/tir/ir/expr_functor.cc:67
  5: VisitExpr
        at /home/xiachunwei/Software/clean_tvm/tvm/src/te/operation/compute_op.cc:545
  4: tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>::VisitExpr(tvm::PrimExpr const&)
        at /home/xiachunwei/Software/clean_tvm/tvm/include/tvm/tir/expr_functor.h:114
  3: tvm::NodeFunctor<void (tvm::runtime::ObjectRef const&, tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>*)>::operator()(tvm::runtime::ObjectRef const&, tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>*) const
        at /home/xiachunwei/Software/clean_tvm/tvm/include/tvm/node/functor.h:97
  2: tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>::InitVTable()::{lambda(tvm::runtime::ObjectRef const&, tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>*)#25}::_FUN(tvm::runtime::ObjectRef const&, tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>*)
        at /home/xiachunwei/Software/clean_tvm/tvm/include/tvm/tir/expr_functor.h:188
  1: tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>::InitVTable()::{lambda(tvm::runtime::ObjectRef const&, tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>*)#25}::operator()(tvm::runtime::ObjectRef const&, tvm::tir::ExprFunctor<void (tvm::PrimExpr const&)>*) const
        at /home/xiachunwei/Software/clean_tvm/tvm/include/tvm/tir/expr_functor.h:188
  0: VisitExpr_
        at /home/xiachunwei/Software/clean_tvm/tvm/src/te/operation/compute_op.cc:551
  File "/home/xiachunwei/Software/clean_tvm/tvm/src/te/operation/compute_op.cc", line 551