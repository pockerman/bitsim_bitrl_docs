\page bitrl_parallel_guide Parallel and distributed training

Reinforcement learning spends most of its time playing episodes, and much of what it plays does not
depend on the rest. bitrl uses that at four levels:

- **many trainings at once**: one training per seed or per setting, each on a thread;
- **many copies of an environment**: one learner plays several episodes at the same time, and learns
  from all of them, as \ref bitrl::rl::algos::pg::ParallelPPOSolver "ParallelPPOSolver" does;
- **an agent per process**: every agent of a multi-agent environment runs in a participant of its own,
  a thread, a process or a machine, with
  \ref bitrl::envs::multi_agent::DistributedAgentEnv "DistributedAgentEnv";
- **a learner over several ranks**: several processes or machines train one network, sharing out the
  episodes and averaging the gradients, with
  \ref bitrl::rl::algos::pg::DistributedPPOSolver "DistributedPPOSolver".

All of it is built on bitsim's thread pool and distributed simulation, see \ref bitsim_parallel_guide.
Every level gives the same result on one thread as on many, to the last bit: the examples check it.

## At a glance

| What | Use | Examples |
| ---- | --- | -------- |
| one training per seed or setting, on all the cores | `bitsim::utils::concurrency::parallel_map` | \ref bitrl_example_multi_agent_grid_world, \ref bitrl_example_multi_robot_navigation |
| several copies of an environment, stepped together | \ref bitrl::envs::ParallelEnvs "ParallelEnvs" | \ref bitrl_example_ppo_parallel_rollouts |
| PPO from several episodes at once | \ref bitrl::rl::algos::pg::ParallelPPOSolver "ParallelPPOSolver" | \ref bitrl_example_ppo_parallel_rollouts |
| several agents in one environment | \ref bitrl::envs::multi_agent::MultiAgentEnvBase "MultiAgentEnvBase", \ref bitrl::rl::multi_agent::MultiAgentTrainer "MultiAgentTrainer" | \ref bitrl_example_multi_agent_grid_world, \ref bitrl_example_multi_robot_navigation |
| every agent in a process of its own | \ref bitrl::envs::multi_agent::DistributedAgentEnv "DistributedAgentEnv", \ref bitrl::rl::multi_agent::AgentTrainer "AgentTrainer" | \ref bitrl_example_distributed_independent_learners, \ref bitrl_example_distributed_robot_navigation |
| PPO over several processes or machines | \ref bitrl::rl::algos::pg::DistributedPPOSolver "DistributedPPOSolver", \ref bitrl::rl::distributed::TCPStoreCommunicator "TCPStoreCommunicator" | \ref bitrl_example_ppo_distributed |

## Many trainings at once

Trainings with different seeds or settings share nothing, so `parallel_map` runs them at the same time
and returns their results in order:

@code{.cpp}
#include "bitsim/utils/concurrency/parallel.h"
using bitsim::utils::concurrency::parallel_map;
using bitsim::utils::concurrency::task_seed;

const auto runs = parallel_map(n_seeds, [](std::size_t k) {
    auto env = make_env();                 // every training its own environment
    Learner learner(task_seed(1, k));      // and its own random generator
    return train(env, learner);
});
@endcode

\ref bitrl_example_multi_agent_grid_world trains 32 teams of independent Q-learners 3.1 times faster on
8 threads, and every team learns exactly what it learns in a plain loop. That needs every training to
have its own environment, policy and random generator: a generator shared between threads is a data
race, and its numbers would depend on the order the threads draw them.

Networks in libtorch draw their initial weights from one generator for the whole process, so build the
networks of the trainings one after the other, before the threads start, and sample actions with a
generator of each training's own (`torch::multinomial(probs, 1, true, generator)`). Small networks run
faster with `torch::set_num_threads(1)`, which keeps libtorch's own threads out of the way of yours.

## Many copies of an environment

\ref bitrl::envs::ParallelEnvs "ParallelEnvs" holds several copies of an environment and steps them
together, on bitsim's thread pool:

@code{.cpp}
#include "bitrl/envs/parallel_envs.h"

ParallelEnvs<CartPole> envs(8);
envs.make("v1", {{"max_steps", uint_t(500)}}, {{"seed", uint_t(42)}}); // copy i: task_seed(42, i)
const auto &first = envs.reset();          // a time step per copy
const auto &next = envs.step(actions);     // an action per copy
@endcode

A copy whose episode is over keeps its last time step, and ignores its action, until the next
`reset()`; `done(i)` and `all_done()` tell which are over. The copies share nothing, so what they do does
not depend on the threads.

Threads pay off for environments whose step takes time: a physics engine, a server over the network.
With an environment that takes 1 ms per step, \ref bitrl_example_ppo_parallel_rollouts plays 8 copies
3.2 times faster on threads. A step of the native CartPole takes about a microsecond, less than handing
it to a thread; step such copies on the calling thread with `use_threads(false)`.

### PPO from several episodes at once

\ref bitrl::rl::algos::pg::ParallelPPOSolver "ParallelPPOSolver" plays one episode in every copy of a
`ParallelEnvs`, choosing the actions of all the copies with one call of the policy, and then makes one
PPO update from all the episodes. The policy needs an `act_batch(states)` that returns the actions and
their log-probabilities for a batch of states.

The gain comes mostly from the batch, not from the threads: one forward pass for 8 states costs about
the same as for one. On CartPole, 800 episodes take 6.7 s, against 38.8 s with
\ref bitrl::rl::algos::pg::PPOSolver "PPOSolver", and both learn to balance the pole for 500 steps.
The episodes do not depend on the number of threads, so a run with the copies on threads learns exactly
what a run without them does.

## Several agents

\ref bitrl::envs::multi_agent::MultiAgentEnvBase "MultiAgentEnvBase" is the base of environments with
several agents that act at the same time: a step takes the joint action, one action per agent, and
returns a \ref bitrl::envs::multi_agent::MultiAgentTimeStep "MultiAgentTimeStep" with an observation
and a reward per agent. \ref bitrl::rl::multi_agent::MultiAgentTrainer "MultiAgentTrainer" trains a team
in it, such as `IndependentQLearning`, a team of
\ref bitrl::rl::multi_agent::QLearningAgent "QLearningAgent"s that each learn from their own
observation.

### Every agent in a process of its own

\ref bitrl::envs::multi_agent::DistributedAgentEnv "DistributedAgentEnv" puts every agent in a
participant of its own: a thread, a process, or a machine such as a robot's on-board computer. Every
participant holds a copy of the whole environment, and only the actions travel:

@code{.cpp}
#include "bitrl/envs/multi_agent/distributed_agent_env.h"

auto copy = make_env();                              // the same options and seed everywhere
QLearningAgent learner(agent, copy.n_states(), copy.n_actions(), config);
bitsim::distributed::MQTT_Transport transport("tcp://localhost:1883", "corridor");
DistributedAgentEnv<MultiAgentGridWorld> env(copy, agent, transport);

AgentTrainer<DistributedAgentEnv<MultiAgentGridWorld>, QLearningAgent> trainer({n_episodes}, learner);
trainer.train(env);
@endcode

`env.step(action)` sends this agent's action, waits for the actions of the others at the same step,
with bitsim's \ref bitsim::distributed::Synchroniser "Synchroniser" in lockstep, and steps the copy with
the joint action. The copies start alike and get the same joint actions, so they stay alike, and every
agent learns exactly what it learns in one process. To make sure, every action comes with a checksum of
the observation it was chosen from, and `step()` throws if that is not the observation in this
participant's copy: a copy made with other options, reset with another seed, or an environment that is
not deterministic.

Distributing has a price: every step waits for a message from every other participant, about 80
microseconds between threads and 130 microseconds through mosquitto with `set_tcp_nodelay`. It pays off
when the agents cost more than that, a learner or a controller per robot, or when the agent must run
where its robot is.

This suits environments that are cheap and deterministic, so that every participant can afford a copy
and the copies agree. A world that is neither, robots in a physics engine for one, belongs in bitsim's
distributed simulation instead, where every participant simulates only its own robots, see
\ref bitsim_parallel_guide.

## PPO over several ranks

\ref bitrl::rl::algos::pg::DistributedPPOSolver "DistributedPPOSolver" trains one network over several
*ranks*: threads, processes or machines. Every rank plays its own episodes, in its own `ParallelEnvs`,
and computes its own gradients; then every rank replaces its gradients with their mean over the ranks,
and takes the same optimiser step. This is the data parallelism of PyTorch's DistributedDataParallel.

@code{.cpp}
#include "bitrl/rl/algorithms/pg/distributed_ppo.h"

TCPStoreCommunicator communicator(host, port, rank, n_ranks);  // rank 0 runs the store
ParallelEnvs<CartPole> envs(n_copies);
envs.make("v1", {{"max_steps", uint_t(500)}}, {{"seed", task_seed(SEED, rank)}}); // other episodes per rank
DistributedPPOSolver<CartPole, Actor, Critic> solver(config, actor, critic,
                                                     actor_optimizer, critic_optimizer,
                                                     communicator);
solver.actions_before_training_begins(envs);                   // every rank takes rank 0's networks
for (uint_t round = 0; round < n_rounds; ++round)
    solver.on_training_episode(envs, round);
@endcode

The ranks add their gradients in the same order, the order of the ranks, so every rank gets the same
mean to the last bit, takes the same step, and keeps the same networks. `checksum()` tells that they do.
Four processes end with exactly the networks of four threads.

The ranks exchange numbers through a \ref bitrl::rl::distributed::Communicator "Communicator".
\ref bitrl::rl::distributed::TCPStoreCommunicator "TCPStoreCommunicator" moves them through
`c10d::TCPStore`, the key-value store that comes with libtorch: rank 0 runs it, and the others connect
to it from this machine or another. Every rank receives every rank's gradients, which is fine for the
networks of RL, with thousands of weights. Networks with millions want a ring all-reduce, such as Gloo's
or MPI's, behind the same interface; libtorch has Gloo compiled in, but does not ship its headers.

What the ranks share is the update, so the gain grows with the episodes per update. On CartPole with 8
episodes per update, 4 ranks take 4.7 s against 5.5 s on one thread; with 32 episodes per update, 4.0 s
against 6.6 s. On one machine, `ParallelPPOSolver` with its copies on threads does much the same. Ranks
pay off when the episodes outgrow one machine, or when every rank runs an expensive simulation of its
own.

## Repeatable runs

Every level above gives the same result whatever the number of threads, if the code follows four rules:

1. **A generator per task**, seeded with `task_seed(seed, task)`: per training, per copy of an
   environment, per agent, per rank.
2. **Results in a fixed order**: `parallel_map` returns them in the order of their index;
   `all_reduce_sum` adds in the order of the ranks.
3. **Nothing shared that changes**: every task its own environment, policy and network.
4. **libtorch's global generator only before the threads start**, to build the networks.

## Which to use

| You want | Use |
| -------- | --- |
| to compare seeds, settings or algorithms | `parallel_map` over the trainings |
| PPO to learn from more episodes per update | `ParallelPPOSolver` |
| an environment whose step is slow to keep the cores busy | `ParallelEnvs` with its copies on threads |
| agents that run on separate computers, or real robots | `DistributedAgentEnv` |
| more episodes per update than one machine can play | `DistributedPPOSolver` |

## The examples

- \ref bitrl_example_ppo_parallel_rollouts
- \ref bitrl_example_ppo_distributed
- \ref bitrl_example_multi_agent_grid_world
- \ref bitrl_example_multi_robot_navigation
- \ref bitrl_example_distributed_independent_learners
- \ref bitrl_example_distributed_robot_navigation
