## TVM lower schedule to Module

入口函数
```C++
IRModule LowerSchedule(te::Schedule sch, const Array<ObjectRef>& args, const std::string& name,
                       const std::unordered_map<te::Tensor, tir::Buffer>& binds,
                       GlobalVarSupply global_var_supply, bool simple_mode) {
  // 1. Lower ComputeOp to TensorIR based on user defined schedule
  IRModule mod = ScheduleToModule(std::move(sch), args, name, binds, global_var_supply);
  // Get the legacy TE pass list
  Array<transform::Pass> pass_list = CreatePassList(simple_mode);
  // 2. Lower passes on TensorIR
  return LowerWithPassList(mod, pass_list);
}
```

### Lower tir Passes
入口代码在driver_api.cc

```C++

Array<tvm::transform::Pass> CreatePassList(bool disable_loop_partition) {
  transform::PassContext pass_ctx = transform::PassContext::Current();

  Array<transform::Pass> user_lower_phase0 = Array<transform::Pass>();
  Array<transform::Pass> user_lower_phase1 = Array<transform::Pass>();
  Array<transform::Pass> user_lower_phase2 = Array<transform::Pass>();
  Array<transform::Pass> user_lower_phase3 = Array<transform::Pass>();

  // Construct the pass list, inserting the user provided passes at the end of the phase

  // PHASE 0
  Array<tvm::transform::Pass> pass_list = user_lower_phase0;

  // PHASE 1
  pass_list.push_back(tir::transform::InjectPrefetch());
  pass_list.push_back(tir::transform::TextureFlatten());
  pass_list.push_back(tir::transform::StorageFlatten(64, instrument_bound_checkers));
  pass_list.push_back(tir::transform::LowerCrossThreadReduction());
  pass_list.push_back(tir::transform::LowerInitBlock());
  pass_list.push_back(tir::transform::PlanAndUpdateBufferAllocationLocation());
  pass_list.push_back(tir::transform::ConvertBlocksToOpaque());
  pass_list.push_back(tir::transform::UnifyThreadBinding());
  pass_list.push_back(tir::transform::ManifestSharedMemoryLocalStage());
  pass_list.push_back(tir::transform::CompactBufferAllocation());
  pass_list.push_back(tir::transform::LowerMatchBuffer());
  pass_list.push_back(tir::transform::InjectSoftwarePipeline());
  pass_list.push_back(tir::transform::LowerOpaqueBlock());
  pass_list.push_back(tir::transform::FlattenBuffer());
  pass_list.push_back(tir::transform::BF16Legalize());
  pass_list.push_back(tir::transform::NarrowDataType(32));
  pass_list.push_back(tir::transform::Simplify());

  // Add user-defined phase-1 passes
  pass_list.insert(pass_list.end(), user_lower_phase1.begin(), user_lower_phase1.end());

  // PHASE 2
  if (!disable_loop_partition) {
    pass_list.push_back(tir::transform::LoopPartition());
  }

  pass_list.push_back(tir::transform::VectorizeLoop(!disable_vectorize));
  pass_list.push_back(tir::transform::InjectVirtualThread());
  pass_list.push_back(tir::transform::InjectDoubleBuffer());
  if (!disable_storage_rewrite) {
    pass_list.push_back(tir::transform::StorageRewrite());
  }
  // LowerVtcmAlloc must occur after any transformations that modify memory allocation locations
  pass_list.push_back(tir::transform::LowerVtcmAlloc());
  bool use_async_copy = pass_ctx->GetConfig<Bool>("tir.use_async_copy", Bool(false)).value();

  if (use_async_copy) {
    pass_list.push_back(tir::transform::LowerAsyncDMA());
  }
  pass_list.push_back(tir::transform::UnrollLoop());

  // Add user-defined phase-2 passes
  pass_list.insert(pass_list.end(), user_lower_phase2.begin(), user_lower_phase2.end());

  // PHASE 3
  pass_list.push_back(tir::transform::RenormalizeSplitPattern());
  pass_list.push_back(tir::transform::Simplify());
  pass_list.push_back(tir::transform::RemoveNoOp());
  pass_list.push_back(tir::transform::RewriteUnsafeSelect());
  pass_list.push_back(tir::transform::HoistIfThenElse());

  // Add user-defined phase-3 passes
  pass_list.insert(pass_list.end(), user_lower_phase3.begin(), user_lower_phase3.end());

  if (instrument_bound_checkers) {
    pass_list.push_back(tir::transform::InstrumentBoundCheckers());
  }

  pass_list.push_back(
      tir::transform::CommonSubexprElimTIR(!disable_cse_tir, enable_equiv_terms_in_cse_tir));

  return pass_list;
}

```

```C++
// implement the provide utility.
Stmt ComputeOpNode::BuildProvide(const Stage& stage,
                                 const std::unordered_map<IterVar, Range>& dom_map,
                                 bool debug_keep_trivial_loop) const {
  ICHECK_EQ(stage->op.operator->(), this);
  ComputeType ctype = DetectComputeType(this, stage);
  if (ctype == ComputeType::kCrossThreadReduction) {
    // specially handle cross thread reduction.
    return MakeCrossThreadReduction(this, stage, dom_map, debug_keep_trivial_loop);
  } else if (ctype == ComputeType::kTensorize) {
    return MakeTensorize(this, stage, dom_map, debug_keep_trivial_loop);
  } else {
    return MakeComputeStmt(this, stage, dom_map, debug_keep_trivial_loop);
  }
}

```