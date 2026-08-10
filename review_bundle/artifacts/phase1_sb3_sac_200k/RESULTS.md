# Standard SB3 SAC Persistent Navigation Baseline

## Protocol

- Stable-Baselines3 SAC 2.8.0, unchanged actor/critic/entropy implementation.
- Seeds: 0, 1, 2.
- Training budget: 200,000 environment steps per seed.
- Persistent open-world random-goal navigation with 5,000-step episodes.
- Held-out evaluation: seeds 100-104, 2,000 deterministic steps each.

## Completion Status

All three runs completed with exit code 0. No NaN, nonfinite update, traceback, or simulator failure was recorded. Checkpoints were produced at 10k, 50k, 100k, and 200k locally. This repository retains only the final 200k checkpoint for each seed.

## Training Results

| Seed | Tasks completed | Tasks / 1k steps | Last-50k tasks / 1k | Collision rate |
|---:|---:|---:|---:|---:|
| 0 | 133 | 0.665 | 1.10 | 0.553% |
| 1 | 206 | 1.030 | 1.42 | 0.785% |
| 2 | 82 | 0.410 | 0.92 | 0.562% |
| Mean | 140.3 | 0.702 | 1.147 | 0.633% |

The full-budget training result confirms that standard SAC can learn task completion in this environment. This is a task-learnability conclusion, not a strict safety or robust-generalization conclusion.

## Held-Out Evaluation at 200k

| Training seed | Completed tasks across 5 evaluations | Successful evaluation streams | Tasks / 1k steps | Collision rate |
|---:|---:|---:|---:|---:|
| 0 | 1 | 1 / 5 | 0.10 | 18.43% |
| 1 | 1 | 1 / 5 | 0.10 | 0.00% |
| 2 | 3 | 2 / 5 | 0.30 | 0.00% |
| Overall | 5 | 4 / 15 | 0.167 | 6.143% |

The aggregate collision count is caused entirely by one pathological evaluation: training seed 0 evaluated on held-out seed 102 produced 1,843 boundary-contact steps out of 2,000. The other 14 held-out streams recorded zero collisions. This outlier must not be hidden by averaging.

From the 10k to 200k checkpoint, mean held-out goal distance decreased from 1.867 to 0.585 and mean minimum goal distance decreased from 0.951 to 0.301. Navigation behavior improved, but deterministic held-out completion remains seed-sensitive.

## Interpretation Boundary

- Classification: `TASK_LEARNABILITY_CONFIRMED` for the training task/distribution.
- Not supported: robust held-out success or safety.
- Not observed: implementation failure, optimization divergence, NaN, or insufficient training budget.
- Safety interpretation: collision telemetry remains a real violation even though collision does not terminate a training episode.
- Comparison caveat: training episodes use 5,000 steps, while each held-out stream uses 2,000 deterministic steps.

## Uploaded Artifact Policy

Tracked artifacts include configs, commands, final summaries, complete learning curves, episode records, held-out evaluations at all checkpoint steps, logs, and the final 200k checkpoints. Intermediate model checkpoints and machine-specific PID/RUNNING files remain local only.
