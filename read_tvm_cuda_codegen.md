## TVM CodeGen 代码阅读

以CUDA为例，入口函数是从`/home/xiachunwei/Projects/intel_tvm/src/target/opt/build_cuda_on.cc`的`BuildCUDA`函数，完成了代码生成和将代码调用NVRTC编译为ptx的过程。

其中`CodeGenCUDA::AddFunction`和`CodeGenCUDA::Finish`是完成了函数的C代码生成。


```C++
runtime::Module BuildCUDA(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  CodeGenCUDA cg;
  cg.Init(output_ssa);

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "CodeGenCUDA: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenCUDA: expect calling_conv equals CallingConv::kDeviceKernelLaunch";
    cg.AddFunction(f);
  }

  std::string code = cg.Finish();

  if (const auto* f = Registry::Get("tvm_callback_cuda_postproc")) {
    code = (*f)(code).operator std::string();
  }
  std::string fmt = "ptx";
  std::string ptx;
  const auto* f_enter = Registry::Get("target.TargetEnterScope");
  (*f_enter)(target);
  if (const auto* f = Registry::Get("tvm_callback_cuda_compile")) {
    ptx = (*f)(code).operator std::string();
    // Dirty matching to check PTX vs cubin.
    // TODO(tqchen) more reliable checks
    if (ptx[0] != '/') fmt = "cubin";
  } else {
    ptx = NVRTCCompile(code, cg.need_include_path());
  }
  const auto* f_exit = Registry::Get("target.TargetExitScope");
  (*f_exit)(target);
  return CUDAModuleCreate(ptx, fmt, ExtractFuncInfo(mod), code);
}
```

调用`CodeGenC::AddFunction`函数是代码生成的核心。
可以看到分别打印函数的声明、函数名、函数的参数、函数体等。

```C++
void CodeGenC::AddFunction(const PrimFunc& f) {
  // clear previous generated state.
  this->InitFuncState(f);
  // reserve keywords
  ReserveKeywordsAsUnique();

  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.defined())
      << "CodeGenC: Expect PrimFunc to have the global_symbol attribute";
  bool no_alias = f->HasNonzeroAttr(tir::attr::kNoAlias);

  this->PrintFuncPrefix(stream);
  this->PrintExtraAttrs(f);
  this->stream << " " << static_cast<std::string>(global_symbol.value()) << "(";

  for (size_t i = 0; i < f->params.size(); ++i) {
    tir::Var v = f->params[i];
    std::string vid = AllocVarID(v.get());
    if (i != 0) stream << ", ";
    if (v.dtype().is_handle()) {
      auto it = alloc_storage_scope_.find(v.get());
      if (it != alloc_storage_scope_.end()) {
        PrintStorageScope(it->second, stream);
      }

      PrintType(GetType(v), stream);
      // Register handle data type
      // TODO(tvm-team): consider simply keep type info in the
      // type annotation(via a normalizing rewriting).
      if (auto* ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (auto* prim = ptr->element_type.as<PrimTypeNode>()) {
          RegisterHandleType(v.get(), prim->dtype);
        }
      }

      if (no_alias) {
        PrintRestrict(v, stream);
      }
    } else {
      PrintType(GetType(v), stream);
    }
    stream << ' ' << vid;
  }
  stream << ") {\n";
  this->PreFunctionBody(f);
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body);
  this->PrintFinalReturn();
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";
}

```

其中核心是函数体的打印。其调用`PrintStmt`函数实际上到了`VisitStmt`: 
```C++
  /*!
   * \brief Print the Stmt n to CodeGenC->stream
   * \param n The statement to be printed.
   */
  void PrintStmt(const Stmt& n) { VisitStmt(n); }`
```


## 如何添加一个新的后端
首先看build为Module的接口：
```C++
runtime::Module build(const Map<String, IRModule>& inputs_arg, const Target& target_host_arg)
 {
  Map<Target, IRModule> updated_inputs;
  Target target_host = target_host_arg;
  for (const auto& it : inputs_arg) {
    Target target = Target(it.first);
    CheckAndUpdateHostConsistency(&target, &target_host);
    Optional<String> device = target->GetAttr<String>("device");
    if (device.defined() && device.value() == "vta") {
      target = Target("ext_dev");
    }
    updated_inputs.Set(target, it.second);
  }
  return TIRToRuntime(updated_inputs, target_host);
}
```


其主要的函数为`TIRToRuntime`。其核心为device进行codegen的: `device_modules.push_back(codegen::Build(device_mod, it.first));`
```C++
runtime::Module TIRToRuntime(const Map<Target, IRModule>& inputs_arg,
                             const Target& target_host_arg) {
  std::vector<runtime::Module> device_modules;
  Map<Target, IRModule> inputs = inputs_arg;
  Target target_host = target_host_arg;

  // Fetch previous defined target host in targets
  CheckAndUpdateHostConsistency(&inputs, &target_host);

  if (!target_host.defined()) {
    target_host = DefaultTargetHost(target_host);
  }

  // Update target host for all targets
  CheckAndUpdateHostConsistency(&inputs, &target_host);

  // Take the attrs from the first module so the eventual modules have them.
  // Ideally this would just be one unified module all the way through;
  IRModule first_module = (*inputs.begin()).second;
  IRModule mhost_all = IRModule(Map<GlobalVar, BaseFunc>(), {}, {}, {}, first_module->attrs);

  ICHECK(mhost_all.defined()) << "The host module must be defined";

  for (const auto& it : inputs) {
    if (it.second.defined()) {
      const Target& target = it.first;
      const IRModule& ir_module = it.second;
      auto pair = SplitMixedModule(ir_module, target, target_host);
      auto& host_mod = pair.first;
      auto& device_mod = pair.second;

      ICHECK(host_mod.defined()) << "The split host module must be defined";

      ICHECK(mhost_all.defined()) << "The host module must be defined";

      // We don't want library modules going back into host codegen
      // unless they're supposed to. Here if we overrode the target host
      // to allow lowering previously we check that it's meant to be placed
      // back into the host Module.
      bool overrides_host_target =
          target->GetTargetDeviceType() == target_host->GetTargetDeviceType();
      bool non_host_target_kind = target->kind != target_host->kind;
      if (overrides_host_target && non_host_target_kind) {
        device_modules.push_back(codegen::Build(host_mod, it.first));
      } else {
        mhost_all->Update(host_mod);
      }

      if (device_mod->functions.size() != 0) {
        device_modules.push_back(codegen::Build(device_mod, it.first));
      }
    }
  }

  runtime::Module mhost = codegen::Build(mhost_all, target_host);
  for (const auto& it : device_modules) {
    if (it.operator->()) {
      mhost.Import(it);
    }
  }

  return mhost;
}
```

进入codegen::Build，从这个地方开始根据各个后端的不同，调用不同的codegen的函数：
```C++
runtime::Module Build(IRModule mod, Target target) {
  if (transform::PassContext::Current()
          ->GetConfig<Bool>("tir.disable_assert", Bool(false))
          .value()) {
    mod = tir::transform::SkipAssert()(mod);
  }

  auto target_attr_map = tvm::TargetKind::GetAttrMap<FTVMTIRToRuntime>("TIRToRuntime");
  if (target_attr_map.count(target->kind)) {
    return target_attr_map[target->kind](mod, target);
  }

  // the build function.
  std::string build_f_name = "target.build." + target->kind->name;
  const PackedFunc* bf = runtime::Registry::Get(build_f_name);
  ICHECK(bf != nullptr) << build_f_name << " is not enabled";
  return (*bf)(mod, target);
}
```
找到`target.build.cuda`