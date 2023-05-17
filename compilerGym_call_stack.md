
# Notes on CompilerGym Source Code Reading

## Default tutorial example
Tutorial code list:

```Python 
import gym
import compiler_gym

env = compiler_gym.make("llvm-v0", 
                        benchmark="cbench-v1/qsort", 
                        observation_space="Autophase",
                        reward_space="IrInstructionCountOz")
env.reset()
# env.render()
observation, reward, done, info = env.step(env.action_space.sample())
print(observation, reward, done, info)
```

For the following statement:
```observation, reward, done, info = env.step(env.action_space.sample())```

The type of `env` is `llvmEnv` which is defined at `CompilerGym/compiler_gym/envs/llvm/llvm_env.py:69`,
`llvmEnv` inherents `ClientServiceCompilerEnv`,
whose's  `service` is defined by the following statements:
```Python
# Start a new service if required.
if self.service is None:
    self.service = CompilerGymServiceConnection(
        self._service_endpoint, self._connection_settings
    )
```
The service is launched by executing the following command:

```shell
./compiler_gym-llvm-service --working_dir=/dev/shm/compiler_gym_xiachunwei/s/0517T172608-178449-9eaa
```

### How to find the source code of `compiler_gym-llvm-service`

`compiler_gym-llvm-service` is defined in `CompilerGym/compiler_gym/envs/llvm/service/BUILD` (details are omitted here, refer to [Bazel: General Rules](https://bazel.build/reference/be/general#filegroup)).
We can naturally infer that `compiler_gym-llvm-service` is the compiled binary of `RunService.cc`.

### How env takes action

The statement 
`const auto ret = createAndRunCompilerGymService<LlvmSession>(argc, argv, usage);` creates an RPC service.
File `/data/xiachunwei/Software/CompilerGym/compiler_gym/service/runtime/CreateAndRunCompilerGymServiceImpl.h:54` shows how the gRPC service is created.
We can use this info to attach to the process for debugging.
In class `CompilerGymService`, the following function defines action:
```C++
grpc::Status Step(grpc::ServerContext* context, const StepRequest* request,
                    StepReply* reply) final override;
```
Which is called in the Python level(More details needs to be added here).
This function is implemented in `CompilerGymServiceImpl.h:138` an is one of the core function of CompilerGym.

```C++

template <typename CompilationSessionType>
grpc::Status CompilerGymService<CompilationSessionType>::Step(grpc::ServerContext* context,
                                                              const StepRequest* request,
                                                              StepReply* reply) {
  CompilationSession* environment;
  RETURN_IF_ERROR(session(request->session_id(), &environment));

  VLOG(2) << "Session " << request->session_id() << " Step()";

  bool endOfEpisode = false;
  std::optional<ActionSpace> newActionSpace;
  bool actionsHadNoEffect = true;

  // Apply the actions.
  for (int i = 0; i < request->action_size(); ++i) {
    bool actionHadNoEffect = false;
    std::optional<ActionSpace> newActionSpaceFromAction;
    RETURN_IF_ERROR(environment->applyAction(request->action(i), endOfEpisode,
                                             newActionSpaceFromAction, actionHadNoEffect));
    ...
  }

  // Compute the requested observations.
  for (int i = 0; i < request->observation_space_size(); ++i) {
    const ObservationSpace* observationSpace;
    RETURN_IF_ERROR(
        observation_space(environment, request->observation_space(i), &observationSpace));
    DCHECK(observationSpace) << "No observation space set";
    RETURN_IF_ERROR(environment->computeObservation(*observationSpace, *reply->add_observation()));
  }

  // Call the end-of-step callback.
  RETURN_IF_ERROR(environment->endOfStep(actionsHadNoEffect, endOfEpisode, newActionSpace));

  reply->set_action_had_no_effect(actionsHadNoEffect);
  if (newActionSpace.has_value()) {
    *reply->mutable_new_action_space() = *newActionSpace;
  }
  reply->set_end_of_session(endOfEpisode);
  return Status::OK;
}
```
For now we don't care about how gRPC works (We can refer to the classical paper [Implementing Remote Procedure Calls](https://web.eecs.umich.edu/~mosharaf/Readings/RPC.pdf) for implementation details).
Let's dive into the `LlvmSession` to see the `applyAction`.

### applyAction in LlvmEnv

In `LlvmSession.cc`, 
```C++
Status LlvmSession::applyAction(const Event& action, bool& endOfEpisode,
                                std::optional<ActionSpace>& newActionSpace,
                                bool& actionHadNoEffect) {
  DCHECK(benchmark_) << "Calling applyAction() before init()";

  // Apply the requested action.
  switch (actionSpace()) {
    case LlvmActionSpace::PASSES_ALL:
      LlvmAction actionEnum;
      if (action.value_case() != Event::ValueCase::kInt64Value) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      fmt::format("Invalid action. Expected {}, received {}.",
                                  magic_enum::enum_name(Event::ValueCase::kInt64Value),
                                  magic_enum::enum_name(action.value_case())));
      }
      RETURN_IF_ERROR(util::intToEnum(action.int64_value(), &actionEnum));
      RETURN_IF_ERROR(applyPassAction(actionEnum, actionHadNoEffect));
  }

  return Status::OK;
}
```
The action is an `Int64` value and then `applyPassAction`:

```C++

Status LlvmSession::applyPassAction(LlvmAction action, bool& actionHadNoEffect) {
...
// Use the generated HANDLE_PASS() switch statement to dispatch to runPass().
#define HANDLE_PASS(pass) actionHadNoEffect = !runPass(pass);
  HANDLE_ACTION(action, HANDLE_PASS)
#undef HANDLE_PASS

  if (!actionHadNoEffect) {
    benchmark().markModuleModified();
  }

  return Status::OK;
}
```
The MACRO `HANDLE_ACTION` (mapping between action number and the llvmPass) is defined in `CompilerGym/compiler_gym/envs/llvm/service/passes/10.0.0/ActionSwitch.h:9`,
and the action statement will be expanded to something like:
`actionHadNoEffect = !runPass((llvm::createAddDiscriminatorsPass()))`.
The corresponding pass is selected according to the action (which is an integer number).
Finally, the LlvmSession will run llvm pass on the benchmark module:
```C++
bool LlvmSession::runPass(llvm::Pass* pass) {
  llvm::legacy::PassManager passManager;
  setupPassManager(&passManager, pass);
  VLOG(1) << "Run pass " << pass->getPassName();
  return passManager.run(benchmark().module());
}
```