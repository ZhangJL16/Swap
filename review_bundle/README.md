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
