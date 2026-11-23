\page bitsim_parallel_guide Parallel and distributed computing

bitsim runs work in parallel at two scales:

- **on the cores of one machine**, with a pool of threads, `bitsim/utils/concurrency`: independent
  runs, such as one simulation per seed, run at the same time, and threads pass data to each other
  through a queue;
- **across processes and machines**, with a distributed simulation, `bitsim/distributed`: every
  *participant* simulates its own robots and tells the others where they are, either in lockstep or
  keeping pace with the wall clock, so that real robots can take part.

bitrl builds on both, see \ref bitrl_parallel_guide.

## At a glance

| What | Use | Header | Examples |
| ---- | --- | ------ | -------- |
| one function per index, on all the cores | \ref bitsim::utils::concurrency::parallel_for "parallel_for" | `bitsim/utils/concurrency/parallel.h` | \ref bitsim_example_thread_pool_basics |
| a result per index, in the order of the indices | \ref bitsim::utils::concurrency::parallel_map "parallel_map" | `bitsim/utils/concurrency/parallel.h` | \ref bitsim_example_parallel_monte_carlo |
| an independent seed per task | \ref bitsim::utils::concurrency::task_seed "task_seed" | `bitsim/utils/concurrency/parallel.h` | \ref bitsim_example_parallel_monte_carlo |
| tasks with futures, on threads of your own | \ref bitsim::utils::concurrency::ThreadPool "ThreadPool" | `bitsim/utils/concurrency/thread_pool.h` | \ref bitsim_example_thread_pool_basics |
| data from one thread to another | \ref bitsim::utils::concurrency::ConcurrentQueue "ConcurrentQueue" | `bitsim/utils/concurrency/concurrent_queue.h` | \ref bitsim_example_producer_consumer |
| robots simulated by separate participants | \ref bitsim::distributed::Synchroniser "Synchroniser" and a \ref bitsim::distributed::Transport "Transport" | `bitsim/distributed/synchroniser.h` | \ref bitsim_example_distributed_robots |
| the other participants' robots in a Chrono world | \ref bitsim::distributed::CHRONO_Ghosts "CHRONO_Ghosts" | `bitsim/distributed/chrono_ghosts.h` | \ref bitsim_example_distributed_chrono_robots |
| a simulation that keeps up with a real robot | `SyncMode::WALL_CLOCK` | `bitsim/distributed/synchroniser.h` | \ref bitsim_example_hardware_in_the_loop_sync |

## On one machine

### Parallel loops

Most code needs only two functions. `parallel_for(begin, end, f)` calls `f(i)` for every index, and
`parallel_map(n, f)` returns the results of `f(0)`, ..., `f(n - 1)`:

@code{.cpp}
#include "bitsim/utils/concurrency/parallel.h"
using namespace bitsim::utils::concurrency;

// one simulation per seed, all at the same time
const auto results = parallel_map(n_runs, [](std::size_t k) {
    std::mt19937 rng(task_seed(42, k));   // a generator per task
    return simulate(rng);                 // shares nothing with the other runs
});
@endcode

- The range is cut into chunks of consecutive indices, one per thread of the pool plus one that the
  calling thread runs itself. Raise `grain`, the fewest indices in a chunk, when `f(i)` is very cheap.
- `parallel_map` returns the results in the order of their index, whatever order the threads finish
  in, so the output is the same on one thread or many.
- `f` must be safe to call from several threads at once: each call should write only to what belongs
  to its index.
- If calls throw, the first exception is rethrown once every chunk is done.
- Called from a task of the pool itself, both run the calls in order on the calling thread, since
  waiting for the pool from inside it could leave no thread to run them.

### Random numbers

A random generator shared between threads is a data race, and seeding one generator per task with
`seed + k` gives generators whose first numbers are alike. `task_seed(seed, k)` mixes the two numbers
with SplitMix64: every task gets an unrelated seed, which depends only on `seed` and `k`. A run is
then the same on any number of threads, as the examples check.

### The thread pool

\ref bitsim::utils::concurrency::ThreadPool "ThreadPool" is a fixed number of threads that run the
tasks given to them, in the order given. `submit(f, args...)` returns a `std::future` for the result,
and an exception the task throws comes out of the future's `get()`. The loops above use
`default_thread_pool()`, which starts on first use with as many threads as the environment variable
`BITSIM_NUM_THREADS` says, or one per core if it is not set:

@code{.sh}
BITSIM_NUM_THREADS=2 ./parallel_monte_carlo   # use two threads, whatever the machine has
@endcode

Pass a pool of your own to `parallel_for` and `parallel_map` to keep some work apart from the rest.

### Passing data between threads

\ref bitsim::utils::concurrency::ConcurrentQueue "ConcurrentQueue" connects a thread that produces
data, such as a sensor driver, to one that consumes it, such as an estimator. `push()` adds to the
queue, `pop()` waits for an item, `pop_for(timeout)` waits at most `timeout`, and `try_pop()` does not
wait. `close()` wakes every waiting consumer: they get what is left, then an empty `std::optional`,
and `push()` returns `false` from then on. bitsim's MQTT subscriber hands its messages over the same
way.

### What pays off

Threads help when the work for an index takes long next to the cost of handing it out, a few
microseconds: a simulation, a training, a planner. \ref bitsim_example_parallel_monte_carlo runs
20000 laps of a robot with noisy odometry 3.7 times faster on 8 threads, and the runs end in the same
place as on one. Work of a few nanoseconds per index gains nothing; give it a larger `grain` or keep it
on one thread.

## Across processes and machines

A simulation of many robots can outgrow one process or one machine, or mix simulated robots with real
ones. In a distributed simulation every participant, a thread, a process or a machine, simulates its
own robots, and every *sync period* of simulated time it sends where its robots are and hears where
the others' are.

### The pieces

| | What it does |
| --- | --- |
| \ref bitsim::distributed::BodyState "BodyState" | where a body is, how it moves, and how big it is |
| \ref bitsim::distributed::StateMessage "StateMessage" | what a participant sends: its bodies at a step, and numbers that are not bodies, in MessagePack |
| \ref bitsim::distributed::Transport "Transport" | how the messages travel: `InProcessBus` between threads, `MQTT_Transport` between processes and machines, `LossyTransport` to add delays and losses in tests |
| \ref bitsim::distributed::Synchroniser "Synchroniser" | every sync period, sends this participant's bodies and returns what it knows of the others |
| \ref bitsim::distributed::CHRONO_Ghosts "CHRONO_Ghosts" | places the other participants' bodies in this participant's Chrono world |

A participant's loop is always the same:

@code{.cpp}
#include "bitsim/distributed/synchroniser.h"
using namespace bitsim::distributed;

SynchroniserConfig config;
config.mode = SyncMode::LOCKSTEP;
config.sync_period = 0.05;               // simulated seconds between two exchanges
config.peers = {"robot_0", "robot_1"};   // the others

MQTT_Transport transport("tcp://localhost:1883", "my_world");
Synchroniser sync(transport, "robot_2", config);
for (uint_t step = 0; step < n_steps; ++step)
{
    const auto &peers = sync.exchange(step, {BodyState::planar("chassis", x, y, yaw, vx, vy, wz)});
    // steer with what peers says of the others, then simulate up to the next exchange
}
sync.leave();
@endcode

### Lockstep or wall clock

`SynchroniserConfig::mode` decides how the participants keep time.

- **Lockstep.** `exchange(step)` returns once every peer has sent the same step. The simulation runs as
  fast as its slowest participant, and it is repeatable: every run gives the same result, however the
  messages were delayed. While it waits, a participant sends its last two steps again every
  `resend_period`, so a lost message or a participant that connects late costs time, not correctness.
  A peer that does not send a step within `timeout` makes `exchange()` throw.
- **Wall clock.** Every participant keeps pace with the wall clock, `real_time_factor` simulated
  seconds per second, and uses the newest state of each peer, moved forward by its age if
  `extrapolate` is set. Nobody waits, so a real robot, which cannot wait, can take part. The price is
  that the result depends on the delays. A state older than `stale_after` is marked stale, and
  `n_overruns()` counts the steps that started late. Across machines, the participants' clocks must
  agree, with NTP for example, since the age of a state comes from its time stamp.

\ref bitsim_example_hardware_in_the_loop_sync runs a simulation next to a robot that reports its pose
with 80 ms of latency: moving the reported pose forward by its age cuts the error of the believed pose
from 6 cm to 1.3 cm on average.

### Robots in Chrono

A participant that simulates its robots in Project Chrono places *ghosts* of the others' bodies in its
world with \ref bitsim::distributed::CHRONO_Ghosts "CHRONO_Ghosts":

@code{.cpp}
CHRONO_Ghosts ghosts(system);
for (uint_t step = 0; ...; ++step)
{
    const auto &peers = sync.exchange(step, {chrono_body_state("chassis", *robot.get_chassis(), size)});
    ghosts.update(peers);   // create, move and remove the ghosts
    // ... step the robot and the system up to the next exchange
}
@endcode

A ghost is a fixed box of the body's size. Its robots' sensors see it and they bump into it, but it
does not push back, because this participant does not simulate it. That suits robots that sense and
avoid each other; robots that push each other belong in one participant. In
\ref bitsim_example_distributed_chrono_robots two robots in two participants drive at each other, and
each stops when its sonar sees the other's ghost.

### Transports

- \ref bitsim::distributed::InProcessBus "InProcessBus" connects participants that are threads of one
  process. It is the quickest way to try a distributed simulation, and what the tests use.
- \ref bitsim::distributed::MQTT_Transport "MQTT_Transport" connects processes on one machine or
  several through an MQTT broker, such as mosquitto, on the topic `bitsim/world/<world>`. It needs
  bitsim built with MQTT. A robot on the network, a Raspberry Pi for example, joins the world by
  publishing `StateMessage::encode()` to the same topic.
- \ref bitsim::distributed::LossyTransport "LossyTransport" wraps another transport and delays or
  drops its messages, to test how a simulation copes with a bad network.

In lockstep every step is a round trip through the broker, so the broker's latency sets the pace.
Mosquitto delays small messages by default (Nagle's algorithm); with `set_tcp_nodelay true` in its
configuration a lockstep step costs about 130 microseconds instead of about 40 ms.

### Which to use

| You want | Use |
| -------- | --- |
| many independent simulations, trainings or experiments | `parallel_map` on one machine |
| robots that only sense and avoid each other, more than one process can simulate | a distributed simulation in lockstep |
| repeatable runs over a network that loses messages | lockstep: the result does not depend on the network |
| real robots alongside simulated ones | a distributed simulation in wall-clock mode |
| robots that push each other, or a world that must be one physical system | one participant |

## The examples

- \ref bitsim_examples_parallel "11. Parallel computing": \ref bitsim_example_thread_pool_basics,
  \ref bitsim_example_producer_consumer, \ref bitsim_example_parallel_monte_carlo
- \ref bitsim_examples_distributed "12. Distributed simulation": \ref bitsim_example_distributed_robots,
  \ref bitsim_example_distributed_chrono_robots, \ref bitsim_example_hardware_in_the_loop_sync
