# Multiworker Persistent Training Proposal

## Goal

Replace periodic exposure resets in a future experiment with multiple independent persistent environment workers feeding one shared PersistentGeneratorSAC learner and replay buffer. This is a proposal only; it is not used by the running corrected-exposure jobs.

## Semantics

Each worker owns its own:

- physical UAV state and RNG;
- pending random goal and task identifier;
- battery, charging, and recovery lifecycle;
- certificate runtime context and authority mode;
- episode/termination accounting.

Each worker changes goal only after genuine task completion. No worker is periodically reset for goal exposure. Workers may reset only under the existing environment termination/truncation policy.

The learner owns one shared:

- actor, twin critics, target critics, and alpha;
- optimizer state and gradient-step counter;
- replay buffer partitioned or tagged by certificate epoch/scenario manifest;
- global interaction/update schedule.

## Advantages

1. **Persistent semantics preserved per worker:** goal, energy, recovery, charging, and departure remain uninterrupted.
2. **Goal diversity without artificial boundaries:** different workers begin from independently seeded certified starts/goals.
3. **Lifecycle diversity:** workers naturally occupy task, backup, and charging modes at different times.
4. **Throughput:** environment collection can overlap when CPU resources permit.
5. **Cleaner final formulation:** training distribution better matches multiple realizations of the persistent mission than fixed-interval resets.

## Biases and risks

1. **Shared-replay off-policy lag:** experience comes from multiple policy ages and worker state distributions.
2. **Worker imbalance:** long recovery or charging episodes can contribute disproportionate data unless sampling is monitored.
3. **Correlation:** synchronous workers updated from the same policy can remain correlated despite different seeds.
4. **Update-ratio ambiguity:** “one step” must mean either one aggregate transition or one transition per worker; comparisons require a fixed total interaction count.
5. **Certificate epochs:** transitions from incompatible manifests/epochs must not mix in a batch under the current replay contract.
6. **Nondeterministic arrival order:** asynchronous queues can make exact replay ordering irreproducible.

## Replay semantics

Every replay item must contain the real same-worker successor. A worker reset must never supply another worker’s or a new rollout’s initial state as `next_observation`.

Required metadata:

```text
worker_id
worker_seed
worker_step
global_collector_step
episode_id
task_id
goal_id
certificate_epoch
terminated
truncated
```

Recommended implementation starts synchronously:

1. Step each worker once with the current actor snapshot.
2. Insert each real transition independently.
3. Reset only workers that naturally terminate/truncate.
4. Perform a fixed number of learner updates per aggregate transition count.
5. Broadcast the updated actor after a deterministic update boundary.

This avoids asynchronous ordering as the first implementation variable.

## Fair baseline requirements

Compare single-worker persistent versus multiworker persistent with:

- identical total environment interactions, not identical wall time;
- identical total gradient updates and update-to-data ratio;
- same seeds mapped to an explicit worker-seed table;
- same replay capacity, batch size, warmup count, actor/critic initialization, reward, entropy, and certificates;
- identical held-out single-worker persistent evaluation;
- reported per-worker and pooled goal/lifecycle distributions.

An additional throughput comparison may hold wall time fixed, but it must be labeled separately from sample efficiency.

## Implementation complexity

| Component | Complexity | Main risk |
|---|---:|---|
| Synchronous vector collector | Medium | preserving per-worker context and deterministic ordering |
| Shared replay metadata | Low-medium | epoch-compatible batch sampling |
| Asynchronous workers | High | policy lag, nondeterminism, process failure recovery |
| GPU learner / CPU collectors | Medium-high | transfer overhead and actor snapshot consistency |
| Checkpoint/resume | High | restoring every worker RNG and lifecycle exactly |

Start with synchronous CPU workers. Do not begin with asynchronous queues.

## Expected compute cost

Atlas construction may dominate worker startup unless immutable certified artifacts are safely shared or cached. Runtime stepping scales approximately with worker count until CPU/memory bandwidth saturates. Learner cost depends on whether the update ratio is fixed per transition. Shared replay memory grows with transition throughput, not worker count directly.

Before implementation, profile:

```text
atlas build time and memory
environment step time
certificate refresh time
learner update time
serialization/copy time
```

## Proposed future gate

Only implement after the corrected exposure results are known. A first pilot should use two or four synchronous workers, equal total interactions, and no algorithm changes. Success evidence should include broader goal exposure, no replay-boundary corruption, unchanged safety gates, and improved held-out persistent task behavior.
