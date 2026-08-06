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
  --seeds 0 1 2 --steps 10000
```

Results are written to `artifacts/comparison/<scenario>/<method>/seed_<n>/` and aggregated under
`artifacts/comparison/aggregate/`. The checked-in first-round matrix is a reduced validation run;
formal paper comparisons require longer training, real calibration, HIL, and deployment evidence.

Checked-in reduced matrix:

```bash
.venv/bin/python scripts/run_comparison.py \
  --methods sac penalty_sac shield_sac generator_sac \
  --scenarios mission_open mission_obstacle mission_narrow mission_energy_tight \
  --seeds 0 1 2 --steps 2000 --warmup-steps 200 --batch-size 64
```
