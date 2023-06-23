## How TVM's meta_scheduler works

### Tutorial example code
First, we still create a simple tutorial code to run meta_schedule's tune:
```Python
def run_ms_tune(work_dir="ms_work_dir"):
  mod = IRModule({})
  A = relay.var('A', shape=(1024, 2048), dtype='float16')
  B = relay.var('A', shape=(512, 2048), dtype='float16')
  my_func = relay.Function([A, B], relay.nn.dense(A, B))
  mod['main'] = my_func
  executor = relay.backend.Executor("graph", {"link-params": True})
  mod = mod.with_attr("executor", executor)
  target = Target("nvidia/nvidia-a100")
  
  database = ms.relay_integration.tune_relay(
      mod=mod,
      params={},
      target=target,
      work_dir=work_dir,
      max_trials_global=1,
      max_trials_per_task=1
  )

if __name__=="__main__":
  run_ms_tune()
```


### Main elements in tuning
In `tvm/python/tvm/meta_schedule/tune.py`,
function `tune_tasks` displays all the important elements involved in the tuning procedure.

#### Builder
Defined in `python/tvm/meta_schedule/builder/builder.py`,
A Builder will build the BuildInput(which is consisted of IRModule, Target and Other params)
and produce a BuildOutput which contains artifact_path and error msg.

#### Runner
Runner takes an RunnerInput which contains artifact_path and device,
and returns RunnerFeature which contains the run_secs and error messages.

#### CostModel
Predict performance according to tir features,
includs three types of model: XGBoost, Mlp and Random.

#### TaskScheduler
Schedule tuning tasks extracted from an Relay model.

#### ModuleEquality
Judge whether two subgraph is equal (Structure Equal or ignore-ndarray) so that duplicated tensor expressions will be tuned only once.


### The detailed auto tuning process
In `python/tvm/meta_schedule/relay_integration.py`,
tune_relay first run relay passes and extract tuning tasks from relay module,
then call func `tune_tasks` in `python/tvm/meta_schedule/tune.py`.
In `meta_schedule.tune.tune_tasks`,
tvm first set up the corresponding configurations (g.g. `max_trials_global`, `builder`, `runner` etc),
then create a task_scheduler `task_scheduler = TaskScheduler.create(task_scheduler)`.
Then task schedule call `tune` which is defined in `python/tvm/meta_schedule/task_scheduler/task_scheduler.py`.
After that call's into a native function `TaskSchedulerTune`, which is defined in `src/meta_schedule/task_scheduler/task_scheduler.cc`.
Note that tasks, space_generator and search strategy are all set in `meta_schedul.relay_integration.extracted_tasks_to_tune_contexts` with the following code:
```Python
def extracted_tasks_to_tune_contexts(
    extracted_tasks: List[ExtractedTask],
    work_dir: str,
    space: SpaceGenerator.SpaceGeneratorType = "post-order-apply",
    strategy: SearchStrategy.SearchStrategyType = "evolutionary",
    num_tuning_cores: Union[Literal["physical", "logical"], int] = "physical",
    seed: Optional[int] = None,
) -> Tuple[List[TuneContext], List[float]]:
    """Convert ExtractedTask to TuneContext.

    Parameters
    ----------
    tasks : List[ExtractedTask]
        The tasks to be converted
    work_dir : str
        The working directory to store logs and databases
    space : SpaceGenerator.SpaceGeneratorType
        The space generator to use.
    strategy : SearchStrategy.SearchStrategyType
        The search strategy to use.
    num_tuning_cores : Union[Literal["physical", "logical"], int]
        The number of CPU cores to use during tuning.
    seed : Optional[int]
        The random seed to use.

    Returns
    -------
    tasks : List[TuneContext]
        The converted tasks
    task_weights : List[float]
        The weights of the tasks
    """
    tasks: List[TuneContext] = []
    task_weights: List[float] = []
    for task, logger, rand_state in zip(
        extracted_tasks,
        get_loggers_from_work_dir(work_dir, [t.task_name for t in extracted_tasks]),
        fork_seed(seed, n=len(extracted_tasks)),
    ):
        tasks.append(
            TuneContext(
                mod=task.dispatched[0],
                target=task.target,
                space_generator=space,
                search_strategy=strategy,
                task_name=task.task_name,
                logger=logger,
                rand_state=rand_state,
                num_threads=num_tuning_cores,
            ).clone()
        )
        task_weights.append(task.weight)
    return tasks, task_weights

```

### TaskScheduler
See `include/tvm/meta_schedule/task_scheduler.h` for detailed explanation.
task_schedule is responsible for fetching a task from task list,
sending it to Builder and then Run the task to get the task's execution latency.
It will save the tuning records to database.

#### Tune
In `task_scheduler.cc`, following is the core function
```C++
void TaskSchedulerNode::Tune(Array<TuneContext> ctxs, Array<FloatImm> task_weights,
                             int max_trials_global, int max_trials_per_task,
                             int num_trials_per_iter, Builder builder, Runner runner,
                             Array<MeasureCallback> measure_callbacks, Optional<Database> database,
                             Optional<CostModel> cost_model) {
  ...
  for (int i = 0; i < n_tasks; ++i) {
    const TuneContext& ctx = ctxs[i];
    double weight = task_weights[i]->value;
    this->tasks_.push_back(TaskRecord(ctx, weight));
    // Get a series of schedule for a Module
    Array<tir::Schedule> design_spaces =
        ctx->space_generator.value()->GenerateDesignSpace(ctx->mod.value());
    TVM_PY_LOG(INFO, ctx->logger) << "Total " << design_spaces.size()
                                  << " design space(s) generated";
    for (int i = 0, n = design_spaces.size(); i < n; ++i) {
      tir::Schedule sch = design_spaces[i];
      tir::Trace trace = sch->trace().value();
      trace = trace->Simplified(true);
      TVM_PY_LOG(INFO, ctx->logger) << "Design space #" << i << ":\n"
                                    << tir::AsTVMScript(sch->mod()) << "\n"
                                    << Concat(trace->AsPython(false), "\n");
    }
    ctx->search_strategy.value()->PreTuning(max_trials_per_task, num_trials_per_iter, design_spaces,
                                            database, cost_model);
  }
  ...
  int num_trials_already = 0;
  for (int task_id; num_trials_already < max_trials_global && (task_id = NextTaskId()) != -1;) {
    ...
    // Build and Run
    if (Optional<Array<MeasureCandidate>> candidates = task->measure_candidates =
            task->ctx->search_strategy.value()->GenerateMeasureCandidates()) {
      int num_candidates = candidates.value().size();
      num_trials_already += num_candidates;
      TVM_PY_LOG(INFO, this->logger) << "Sending " << num_candidates << " sample(s) to builder";
      SendToBuilder(task, builder);
      TVM_PY_LOG(INFO, this->logger) << "Sending " << num_candidates << " sample(s) to runner";
      SendToRunner(task, runner);
    } else {
      TerminateTask(task_id);
    }
  }
  for (int task_id = 0; task_id < n_tasks; ++task_id) {
    TaskRecordNode* task = this->tasks_[task_id].get();
    if (!task->is_terminated) {
      if (task->runner_futures.defined()) {
        JoinRunningTask(task_id);
      }
      TerminateTask(task_id);
    }
    task->ctx->search_strategy.value()->PostTuning();
  }
}

```

### Dive into SpaceGenerator
The following statement is the core operation to generate schedule for the give `ctx->mod`:
```C++
// Get a series of schedule for a Module
    Array<tir::Schedule> design_spaces =
        ctx->space_generator.value()->GenerateDesignSpace(ctx->mod.value());
```
.
For relay, meta_schedule uses `post-order-apply` space generator to generate schedule.
Therefore, the previous statement calls into `PostOrderApplyNode::PostOrderApplyNode` which is defined in `post_order_apply.cc`.
The `PostOrderApplyNode` generator iteratively apply the schedule rule defined in `sch_rules`.
The `sch_rules` is initialized in `SpaceGeneratorNode::InitializeWithTuneContext` which create default rules for each kind of target.

The `space_generator` in `ctx` is responsible for generating the schedule.
The context would 
Generating many schedules and record trace for an IRModule.
`src/meta_schedule/space_generator/space_generator.cc` defines some default schedule
for each kind of target (like `llvm` or `cuda`).
In `src/meta_schedule/schedule_rule/schedule_rule.cc`,
we can set ScheduleRule for CUDA.
For example, we can set `thread_extends_` and `max_threads_per_block`.


#### MultiSizeTiling


### Evolutional Search

#### Random Sample

#### Pick from database

#### Population

### Cost Model

#### Feature extractor

#### XGBoost Model

#### Predict & Update

in `src/meta_schedule/search_strategy/evolutionary_search.cc:539`,
`std::vector<Schedule> EvolutionarySearchNode::State::EvolveWithCostModel(
std::vector<Schedule> population, int num)` 
use code model to predict scores for the given population.



### Q & A
#### Question 1:
How the schedules are generated

#### How each schedule primitive (aka `Instructions`) is choosed?

#### How to extract features from a scheduled IRModule?

#### How tuning records are saved?

Following is the call stack:
```C++
#0  tvm::meta_schedule::TuningRecordNode::AsJSON (this=0x1d26160)
    at /home/xiachunwei/Software/tvm_oraa/src/meta_schedule/database/database.cc:97
#1  0x00007fffe3cabc36 in tvm::meta_schedule::JSONDatabaseNode::CommitTuningRecord (
    this=0x1c2d550, record=...)
    at /home/xiachunwei/Software/tvm_oraa/src/meta_schedule/database/json_database.cc:116
#2  0x00007fffe3cf3f91 in tvm::meta_schedule::AddToDatabaseNode::Apply (
    this=0x1cc1c80, task_scheduler=..., task_id=0, measure_candidates=..., 
    builder_results=..., runner_results=...)
    at /home/xiachunwei/Software/tvm_oraa/src/meta_schedule/measure_callback/add_to_database.cc:49
#3  0x00007fffe3e5ec71 in tvm::meta_schedule::TaskSchedulerNode::JoinRunningTask (
    this=0x1cca0c0, task_id=0)
    at /home/xiachunwei/Software/tvm_oraa/src/meta_schedule/task_scheduler/task_scheduler.cc:232
#4  0x00007fffe3e54fd9 in tvm::meta_schedule::GradientBasedNode::JoinRunningTask (
    this=0x1cca0c0, task_id=0)
    at /home/xiachunwei/Software/tvm_oraa/src/meta_schedule/task_scheduler/gradient_based.cc:126
#5  0x00007fffe3e549b9 in tvm::meta_schedule::GradientBasedNode::NextTaskId (
    this=0x1cca0c0)
    at /home/xiachunwei/Software/tvm_oraa/src/meta_schedule/task_scheduler/gradient_based.cc:72
#6  0x00007fffe3e5d8f4 in tvm::meta_schedule::TaskSchedulerNode::Tune (
    this=0x1cca0c0, ctxs=..., task_weights=..., max_trials_global=2000, 
    max_trials_per_task=2000, num_trials_per_iter=64, builder=..., runner=..., 
    measure_callbacks=..., database=..., cost_model=...)
--Type <RET> for more, q to quit, c to continue without paging--
   tvm_oraa/src/meta_schedule/task_scheduler/task_scheduler.cc:179
#7  0x00007fffe3e5471f in tvm::meta_schedule::GradientBasedNode::Tune (this=0x1cca0c0, tasks=..., task_weights=..., 
    max_trials_global=2000, max_trials_per_task=2000, num_trials_per_iter=64, builder=..., runner=..., 
    measure_callbacks=..., database=..., cost_model=...)
```
