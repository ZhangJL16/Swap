# Certified UAV Review Bundle

Synthetic single-UAV certificate experiments, multi-step Generator-SAC, and legacy regression
fixtures. This repository is software and synthetic empirical evidence only; it is not real-flight
safety evidence.

## Setup

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## Tests

```bash
.venv/bin/python -m unittest discover -v -s tests -p 'test_*.py'
```

## Multi-Step Training

```bash
.venv/bin/python scripts/train_generator_sac.py \
  --scenario mission_open --seeds 0 1 2 --steps 10000

.venv/bin/python scripts/run_comparison.py \
  --methods sac penalty_sac shield_sac generator_sac \
  --scenarios mission_open mission_obstacle mission_narrow mission_energy_tight \
  --seeds 0 1 2 3 4 --steps 10000 --warmup-steps 1000 \
  --batch-size 128 --evaluation-episodes 20
```

Results are written to `artifacts/comparison/<scenario>/<method>/seed_<n>/` and aggregated under
`artifacts/comparison/aggregate/`. The checked-in 10k matrix is synthetic comparative validation,
not convergence, physical calibration, HIL, WCET, or real-flight evidence.

Before comparison, run the theorem-facing synthetic mission gate:

```bash
.venv/bin/python scripts/validate_mission_recovery_certificate.py
```

The current gate passes all four mission fixtures. In the checked-in 80-run 10k matrix, the two
certified methods have zero sampled collision, uncertified publication, and invalid-κ fallback.
Generator-SAC completes task and return in open/obstacle; narrow and energy-tight deliberately
trigger certified return before task completion. See `docs/SINGLE_UAV_CERTIFICATION_ENV.md` for
scope boundaries and exact aggregate metrics.

Checked-in reduced matrix:

```bash
.venv/bin/python scripts/run_comparison.py \
  --methods sac penalty_sac shield_sac generator_sac \
  --scenarios mission_open mission_obstacle mission_narrow mission_energy_tight \
  --seeds 0 1 2 --steps 2000 --warmup-steps 200 --batch-size 64
```

## RL-Contribution and Generalization Evaluation

The task-oriented Generator is evaluated as a certified reference plus a learned residual. The
non-learning `center_only` and `random_generator` methods share the same mission manifest,
complete-set verifier, and independently certified κ as Generator-SAC.

```bash
.venv/bin/python scripts/run_rl_contribution_ablation.py \
  --scenarios mission_open mission_obstacle --seeds 0 1 2 3 4 --episodes 20
.venv/bin/python scripts/run_center_mode_ablation.py \
  --scenarios mission_open mission_obstacle --seeds 0 1 2 3 4
.venv/bin/python scripts/generate_scenario_families.py \
  --training 20 --validation 10 --heldout 20 --validate-certificates
.venv/bin/python scripts/evaluate_heldout_generalization.py \
  --scenario-index artifacts/scenario_families/scenario_index.json
```

Held-out evaluation refuses missing checkpoints, scenario mutations, failed certificate gates, or
manifest mismatches. Multi-scenario training is available through
`scripts/train_multiscenario_generator.py`, with one immutable scenario/manifest per episode and
replay grouped by manifest epoch. Its default 50k budget is not launched merely because the
interface exists. Results under `artifacts/paper/` remain synthetic empirical evidence.

The completed five-seed, 20-episode-per-seed ablation gives task/return success 1.0 and sampled
collision 0 for Center-Only, Random-in-Generator, and Generator-SAC in both open and obstacle
missions. Relative to Center-Only, Generator-SAC changes mission length by 0 steps in open and
-0.15 steps in obstacle; path differences are approximately -0.000001 m and -0.000397 m. These
gains are negligible at the fixture scale. Current task success is therefore attributed primarily
to the verified task-oriented center, not SAC residual learning.

The held-out pilot rebuilds and validates 20 manifests and evaluates 20 episodes per scenario
using the available seed-0 frozen checkpoints. `GENERALIZATION_GATE` passes: all 60 method-scenario
rows have zero sampled certified-method collision, zero uncertified publication, and zero invalid-
kappa fallback. Generator-SAC and Center-Only both succeed on every open/obstacle held-out mission,
both fail the task in narrow missions while returning successfully, and both reach only 0.20 task
success in energy-tight missions. This is single-checkpoint-seed synthetic evidence and does not
justify a physical or multi-seed generalization claim.

```bash
.venv/bin/python scripts/aggregate_generalization.py
.venv/bin/python scripts/profile_certificate_scalability.py
.venv/bin/python scripts/run_certificate_sensitivity.py --scenario mission_open --episodes 5
```

The post-change regression record is 128/128 tests. A 50k multi-scenario entry point is present,
but larger training is intentionally deferred until a certified family demonstrates room for the
learned residual to improve over Center-Only.

## Persistent Goals and Autonomous Charging

The separate persistent path uses one continuous three-dimensional Generator-SAC policy for task
flight, voluntary station approach, charger dwell, and departure. Every accepted action set must
preserve certified kappa recoverability; kappa is backup authority only. The task center defaults to
`safety_neutral`, pending goals survive charging, and the synthetic charger uses `30.0` capacity,
`2.0` units/s, and `0.4` per 0.2 s. Closed departure uses a complete charger-stay Generator support,
and replay/Bellman targets consume the same immutable execution-authority classification as runtime.
Persistent validation uses typed task/departure/recovery gates: recovery-only cells require the
certified kappa chain but not a Generator, while task-RL roots retain goal/station directional checks.
Shared bound versions are graph-wide; edge-local geometry/kappa/manifest identities remain distinct
and are hash-bound into the aggregate manifest. The corrected synthetic gate run passes
`persistent_open` and `persistent_energy_tight`; `persistent_obstacle` remains blocked by complete
swept-geometry failures in four recovery chains. Acceptance and training were not run.

```bash
.venv/bin/python scripts/validate_persistent_certificate.py
.venv/bin/python scripts/run_persistent_env_acceptance.py --scenario persistent_open --probe all --strict
.venv/bin/python scripts/train_persistent_generator_sac.py --scenario persistent_open --legacy-fixed-graph --steps 50000
.venv/bin/python scripts/evaluate_persistent_generator_sac.py --scenario persistent_open --legacy-fixed-graph --checkpoint <path>
.venv/bin/python scripts/run_persistent_single_policy_baselines.py --scenario persistent_open --legacy-fixed-graph
```

See `docs/PERSISTENT_TASK_CHARGING.md`. These commands remain synthetic and do not provide real
calibration, HIL, hard WCET, or real-flight safety evidence.

### Task-independent random persistent main path

The current main persistent environment is built with
`make_random_persistent_uav_env()`. It samples the initial physical state from a frozen
`CertifiedRecoverabilityAtlas` and samples continuous horizontal goals from certified atlas
interiors. The atlas uses geometry, terminal, dynamics, tracking, energy, and frozen-kappa
evidence only; it does not consume a task edge, task waypoint, goal ID, route index, or task
reward. Thus `c(x), G(x)` are invariant to a goal change at the same certificate state, while the
goal-conditioned actor may change its latent.

The old `persistent_open`, `persistent_obstacle`, and `persistent_energy_tight` graphs remain
legacy certificate-regression fixtures. New main fixtures are `random_persistent_open`,
`random_persistent_obstacle`, and `random_persistent_energy_tight`.

```bash
cd review_bundle
PYTHONPATH=. /home/zjl/mappo/.venv/bin/python scripts/validate_random_persistent_architecture.py
PYTHONPATH=. /home/zjl/mappo/.venv/bin/python scripts/validate_random_persistent_authority_lifecycle.py
PYTHONPATH=. /home/zjl/mappo/.venv/bin/python scripts/train_persistent_generator_sac.py --scenario random_persistent_open --steps 5000
```

This validator is deterministic synthetic software evidence, not training evidence or a
real-flight safety claim.
