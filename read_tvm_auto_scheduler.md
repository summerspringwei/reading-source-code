## 调HostCompileError的bug

在auto_schedule.cc中的AutoSchedule函数为自动调度的入口。
由于error_no是在有两个地方MeasureResult更新了，所以从这两个函数跟下去：
`program_cost_model->Update(inputs, results);`
`results = measurer->Measure(search_task, GetRef<SearchPolicy>(this), inputs);`
Update函数是根据cost_model来预测执行时间。
```C++

State SketchPolicyNode::Search(int n_trials, int early_stopping, int num_measure_per_iter,
                               ProgramMeasurer measurer) {
  num_measure_per_iter_ = num_measure_per_iter;

  if (n_trials <= 1) {
    // No measurement is allowed
    const Array<State>& best_states = SearchOneRound(0);
    ICHECK_GT(best_states.size(), 0);
    return best_states[0];
  } else {
    int num_random =
        static_cast<int>(GetDoubleParam(params, SketchParamKey::eps_greedy) * num_measure_per_iter);
    early_stopping = early_stopping < 0 ? std::numeric_limits<int>::max() >> 1 : early_stopping;
    measurer->Reset();

    int ct = 0;
    int empty_retry_count = GetIntParam(params, SketchParamKey::empty_retry_count);
    Array<State> best_states, random_states;
    Array<MeasureInput> inputs;
    Array<MeasureResult> results;
    while (ct < n_trials) {
      if (!inputs.empty()) {
        auto t_begin = std::chrono::high_resolution_clock::now();

        // Retrain the cost model before the next search round
        PrintTitle("Train cost model", verbose);
        program_cost_model->Update(inputs, results);

        PrintTimeElapsed(t_begin, "training", verbose);
      }

      // Search one round to get promising states
      PrintTitle("Search", verbose);
      best_states = SearchOneRound(num_random * 3, &random_states);

      // Infer bound. This is necessary for computing the correct ToStr() for redundancy check
      best_states = search_task->compute_dag.InferBound(best_states);
      random_states = search_task->compute_dag.InferBound(random_states);

      // Pick `num_measure_per_iter` states to measure, check hash to remove already measured state
      // Also pick some random states to do eps-greedy
      inputs = PickStatesWithEpsGreedy(best_states, random_states, n_trials - ct);

      // Currently it's hard to detect if all of the search space has been traversed
      // Stop if no extra valid states found in several retries
      if (inputs.empty()) {
        if (empty_retry_count-- > 0) {
          continue;
        } else {
          StdCout(verbose) << "It seems all candidates in the search space have been measured."
                           << std::endl;
          break;
        }
      } else {
        // Reset the retry count
        empty_retry_count = GetIntParam(params, SketchParamKey::empty_retry_count);
      }

      // Measure candidate states
      PrintTitle("Measure", verbose);
      results = measurer->Measure(search_task, GetRef<SearchPolicy>(this), inputs);
      ct += inputs.size();

      // Check if reach the early stopping condition
      if (ct - measurer->best_ct[search_task->workload_key] > early_stopping &&
          measurer->has_valid.count(search_task->workload_key)) {
        StdCout(verbose) << "Stop early since no performance improvement in the last "
                         << early_stopping << " measurements trials.\n";
        break;
      }

      // Update measured states throughputs. These states will join the EvolutionarySearch in later
      // search rounds.
      for (const auto& res : results) {
        measured_states_throughputs_.push_back(1.0 / FloatArrayMean(res->costs));
      }
    }
    PrintTitle("Done", verbose);

    return measurer->best_state[search_task->workload_key];
  }
}

```



```C++
Array<MeasureResult> ProgramMeasurerNode::Measure(const SearchTask& task,
                                                  const SearchPolicy& policy,
                                                  const Array<MeasureInput>& inputs,
                                                  int batch_size) {
  auto t_begin = std::chrono::high_resolution_clock::now();

  Array<MeasureResult> results;
  results.reserve(inputs.size());

  if (batch_size == -1) {
    // set default batch size
    batch_size = builder->n_parallel * 2;
  }

  int old_verbosity = verbose;

  StdCout(verbose) << "Get " << inputs.size() << " programs to measure:" << std::endl;

  for (size_t i = 0; i < inputs.size(); i += batch_size) {
    Array<MeasureInput> input_batch(inputs.begin() + i,
                                    inputs.begin() + std::min(i + batch_size, inputs.size()));
    Array<MeasureResult> result_batch;

    // build and run
    SilentMeasure(task, input_batch, &result_batch);

    // update current best state according to the new measure result
    for (size_t j = 0; j < input_batch.size(); ++j) {
      const String& workload_key = input_batch[j]->task->workload_key;
      double flops;

      if (result_batch[j]->error_no == 0) {
        flops = task->compute_dag->flop_ct / FloatArrayMean(result_batch[j]->costs);
        error_ct = 0;
        has_valid.insert(workload_key);
      } else {
        flops = 0.0;
        error_ct++;
      }

      if (flops > best_flops[workload_key]) {
        best_flops[workload_key] = flops;
        best_state[workload_key] = input_batch[j]->state;
        best_ct[workload_key] = ct;
      }

      ct++;
      StdCout(verbose, 2) << std::fixed << std::setprecision(2) << Chars('=', 50) << "\n"
                          << "No: " << ct << "\tGFLOPS: " << flops / 1e9 << " / "
                          << best_flops[workload_key] / 1e9 << "\tresults: " << result_batch[j]
                          << "\n"
                          << Chars('=', 50) << "\n"
                          << input_batch[j]->state << "\n";
    }

    // Call callback functions
    if (callbacks) {
      for (const auto& callback : callbacks.value()) {
        callback->Callback(policy, input_batch, result_batch);
      }
    }

    // Store result batch
    for (auto& res : result_batch) {
      results.push_back(res);
    }

    if (error_ct > max_continuous_error) {
      LOG(WARNING) << "Too many errors happened during tuning. Switching to debug mode."
                   << std::endl;
      verbose = 2;
    } else {
      verbose = old_verbosity;
    }
  }

  PrintTimeElapsed(t_begin, "measurement", verbose);

  return results;
}

```

下面到SilenceMeasure中
```C++
void ProgramMeasurerNode::SilentMeasure(const SearchTask& task, const Array<MeasureInput>& inputs,
                                        Array<MeasureResult>* results) {
  results->clear();
  results->reserve(inputs.size());

  // Call builder and runner
  Array<BuildResult> build_res_batch = builder->Build(inputs, verbose);
  for(auto& br: build_res_batch){
    VLOG(2) << br;
  }
  Array<MeasureResult> result_batch = runner->Run(inputs, build_res_batch, verbose);
  
  // Store result batch
  for (auto& res : result_batch) {
    VLOG(2) << res;
    results->push_back(res);
  }
}
```

然后是builder->Build中，
```C++
Array<BuildResult> LocalBuilderNode::Build(const Array<MeasureInput>& inputs, int verbose) {
  if (const auto* f = runtime::Registry::Get("auto_scheduler.local_builder.build")) {
    Array<BuildResult> results = (*f)(inputs, timeout, n_parallel, build_func, verbose);
    return results;
  }
  LOG(FATAL) << "auto_scheduler.local_builder.build is not registered. "
             << "This is a function registered in Python, "
             << "make sure the TVM Python runtime has been loaded successfully.";
  throw;
}

```
然后这个就到Python中了，在measure.py中
```Python

@tvm._ffi.register_func("auto_scheduler.local_builder.build")
def local_builder_build(inputs, timeout, n_parallel, build_func="default", verbose=1):
    """
    Build function of LocalBuilder to build the MeasureInputs to runnable modules.

    Parameters
    ----------
    inputs : List[MeasureInput]
        The MeasureInputs to be built.
    timeout : int
        The timeout limit (in second) for each build thread.
        This is used in a wrapper of the multiprocessing.Process.join().
    n_parallel : int
        Number of threads used to build in parallel.
    build_func : str = 'default'
        The name of build function to process the built module.
    verbose: int = 1
        Verbosity level. 0 for silent, 1 to output information during program building.

    Returns
    -------
    res : List[BuildResult]
        The build results of these MeasureInputs.
    """
    assert build_func == BuildFunc.name, (
        "BuildFunc.name: " + BuildFunc.name + ", but args is: " + build_func
    )
    executor = PopenPoolExecutor(
        n_parallel, timeout, reset_global_scope, (AutotvmGlobalScope.current,)
    )
    tuple_res = executor.map_with_error_catching(
        local_build_worker,
        [
            (
                i.serialize(),
                BuildFunc.build_func,
                verbose,
            )
            for i in inputs
        ],
    )

    results = []
    for res in tuple_res:
        if res.status == StatusKind.COMPLETE:
            results.append(BuildResult(*res.value))
        elif res.status == StatusKind.TIMEOUT:
            if verbose >= 1:
                print(".T", end="", flush=True)  # Build timeout
            results.append(BuildResult(None, [], MeasureErrorNo.BUILD_TIMEOUT, None, timeout))
        elif res.status == StatusKind.EXCEPTION:
            if verbose >= 1:
                print(".E", end="", flush=True)  # Build error
            results.append(
                BuildResult(None, [], MeasureErrorNo.COMPILE_HOST, repr(res.value), timeout)
            )
        else:
            raise ValueError("Result status is not expected. Unreachable branch")

    return results

```
之后报错的地方就在这里
```Python

def _local_build_worker(inp_serialized, build_func, verbose):
    tic = time.time()
    inp = MeasureInput.deserialize(inp_serialized)
    task = inp.task
    task.target, task.target_host = Target.check_and_update_host_consist(
        task.target, task.target_host
    )

    error_no = MeasureErrorNo.NO_ERROR
    error_msg = None
    args = []

    try:
        sch, args = task.compute_dag.apply_steps_from_state(
            inp.state, layout_rewrite=task.layout_rewrite_option
        )
    # pylint: disable=broad-except
    except Exception:
        error_no = MeasureErrorNo.INSTANTIATION_ERROR
        error_msg = make_traceback_info()

    if error_no == 0:
        dirname = tempfile.mkdtemp()
        filename = os.path.join(dirname, "tmp_func." + build_func.output_format)

        try:
            with transform.PassContext():
                func = build_module.build(sch, args, target=task.target)
            func.export_library(filename, build_func)
        # pylint: disable=broad-except
        except Exception:
            error_no = MeasureErrorNo.COMPILE_HOST
            error_msg = make_traceback_info()
            print(error_msg)
    else:
        filename = ""

    if verbose >= 1:
        if error_no == MeasureErrorNo.NO_ERROR:
            print(".", end="", flush=True)
        else:
            print(".E", end="", flush=True)  # Build error

    return filename, args, error_no, error_msg, time.time() - tic

```