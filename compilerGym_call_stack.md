
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
The `step` function in Python is implemented in `client_service_compiler_env.py:811`.
The action is packed in `request` and call the LlvmService through gRPC with the following statement:
` reply = _wrapped_step(self.service, request, timeout)`.

```Python
def raw_step(
    self,
    actions: Iterable[ActionType],
    observation_spaces: List[ObservationSpaceSpec],
    reward_spaces: List[Reward],
    timeout: float = 300,
) -> StepType:
    """Take a step.

    :param actions: A list of actions to be applied.

    :param observations: A list of observations spaces to compute
        observations from. These are evaluated after the actions are
        applied.

    :param rewards: A list of reward spaces to compute rewards from. These
        are evaluated after the actions are applied.

    :return: A tuple of observations, rewards, done, and info. Observations
        and rewards are lists.

    :raises SessionNotFound: If :meth:`reset()
        <compiler_gym.envs.ClientServiceCompilerEnv.reset>` has not been called.

    .. warning::

        Don't call this method directly, use :meth:`step()
        <compiler_gym.envs.ClientServiceCompilerEnv.step>` or :meth:`multistep()
        <compiler_gym.envs.ClientServiceCompilerEnv.multistep>` instead. The
        :meth:`raw_step() <compiler_gym.envs.ClientServiceCompilerEnv.step>` method is an
        implementation detail.
    """
    ...

    # Record the actions.
    self._actions += actions

    # Send the request to the backend service.
    request = StepRequest(
        session_id=self._session_id,
        action=[
            self.service_message_converters.action_converter(a) for a in actions
        ],
        observation_space=[
            observation_space.index for observation_space in observations_to_compute
        ],
    )
    try:
        reply = _wrapped_step(self.service, request, timeout)
    ...

    # Get the user-requested observation.
    observations: List[ObservationType] = [
        computed_observations[observation_space_index_map[observation_space]]
        for observation_space in observation_spaces
    ]

    # Update and compute the rewards.
    rewards: List[RewardType] = []
    for reward_space in reward_spaces:
        reward_observations = [
            computed_observations[
                observation_space_index_map[
                    self.observation.spaces[observation_space]
                ]
            ]
            for observation_space in reward_space.observation_spaces
        ]
        rewards.append(
            float(
                reward_space.update(actions, reward_observations, self.observation)
            )
        )

    info = {
        "action_had_no_effect": reply.action_had_no_effect,
        "new_action_space": reply.HasField("new_action_space"),
    }

    return observations, rewards, reply.end_of_session, info
```

### How to find the source code of `compiler_gym-llvm-service`

`compiler_gym-llvm-service` is defined in `CompilerGym/compiler_gym/envs/llvm/service/BUILD` (details are omitted here, refer to [Bazel: General Rules](https://bazel.build/reference/be/general#filegroup)).
We can naturally infer that `compiler_gym-llvm-service` is the compiled binary of `RunService.cc`.

### How env takes action

The statement 
`const auto ret = createAndRunCompilerGymService<LlvmSession>(argc, argv, usage);` creates an RPC service.
File `CompilerGym/compiler_gym/service/runtime/CreateAndRunCompilerGymServiceImpl.h:54` shows how the gRPC service is created.
We can use this info to attach to the process for debugging.
In class `CompilerGymService`, the following function defines action:
```C++
grpc::Status Step(grpc::ServerContext* context, const StepRequest* request,
                    StepReply* reply) final override;
```
Which is called in the Python level(More details needs to be added here).
This function is implemented in `CompilerGymServiceImpl.h:138` and is one of the core function of CompilerGym.

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

### Get observation from Env (Get state-transition)

`CompilerGym`  service call computeObservation to get the reward and state-transition.
```C++
Status LlvmSession::computeObservation(const ObservationSpace& observationSpace,
                                       Event& observation) {
  DCHECK(benchmark_) << "Calling computeObservation() before init()";

  const auto& it = observationSpaceNames_.find(observationSpace.name());
  ...
  const LlvmObservationSpace observationSpaceEnum = it->second;

  return setObservation(observationSpaceEnum, workingDirectory(), benchmark(), observation);
}
```
The `ObservationSpaces` is defined in `ObservationSpaces.h:24` and we can find `AutoPhase` here.
They reveal the features of LLVM IR from some aspects like text size or runtime on a specific hard.
Then `LlvmSession` call the `setObservation` (Implemented in `Observation.cc:40`) to get the observation.
The observation is consisted with reward, state-transition and
other customized info and packed in `Event& reply`.
The `relay` will be sent back to the Python level by the gRPC service.
The function `setObservation` will compute the corresponding observation based on the `LlvmObservationSpace`.
Here we list the `AutoPhase`:
```C++
Status setObservation(LlvmObservationSpace space, const fs::path& workingDirectory,
                      Benchmark& benchmark, Event& reply) {
  switch (space) {
    ...
    case LlvmObservationSpace::AUTOPHASE: {
      const auto features = autophase::InstCount::getFeatureVector(benchmark.module());
      *reply.mutable_int64_tensor()->mutable_shape()->Add() = features.size();
      *reply.mutable_int64_tensor()->mutable_value() = {features.begin(), features.end()};
      break;
    }
    ...
    return Status::OK;
  }

```
We can see that the `AutoPhase` get the feature vector from the module's LLVM IR, and pack the feature to `relay`.
Now we have figure it out how a step is applied in `LlvmEnv`.
Let's go back to Python code.

### Compute the reward
The reward is update in the `raw_step` function with the following statements:

```Python
# Update and compute the rewards.
rewards: List[RewardType] = []
for reward_space in reward_spaces:
    reward_observations = [
        computed_observations[
            observation_space_index_map[
                self.observation.spaces[observation_space]
            ]
        ]
        for observation_space in reward_space.observation_spaces
    ]
    rewards.append(
        float(
            reward_space.update(actions, reward_observations, self.observation)
        )
    )
```
The `reward_observation_spaces` is of type `ObservationSpaceSpec(IrInstructionCountOz)`(defined in `observation_space_spec.py:14` and instanized in `llvm_env.py:127 BaselineImprovementNormalizedReward`).
The `reward_space` is `IrInstructionCountOz` (),
which contains `benchmark` and `cost_function`.
The `update` function is defined in class `BaselineImprovementNormalizedReward` at `llvm_reward.py:74`:

```Python
def update(
    self,
    actions: List[ActionType],
    observations: List[ObservationType],
    observation_view: ObservationView,
) -> RewardType:
    """Called on env.step(). Compute and return new reward."""
    if self.cost_norm is None:
        self.cost_norm = self.get_cost_norm(observation_view)
    return super().update(actions, observations, observation_view) / self.cost_norm
```

**Now we have figured out all the four key elements in CompilerGym's source code, which are**
- `Observation space: A sequence of llvm pass`, 
- `Action: apply the selected llvm pass`, 
- `State transition:  Compiling benchmark's module IR to transformed IR by a pass` , 
- `Reward: normalized reduction in number of instructions comparing with Oz`. 

**Cheers!**
