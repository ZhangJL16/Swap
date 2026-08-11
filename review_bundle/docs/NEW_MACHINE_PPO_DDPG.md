# New-machine PPO and DDPG runbook

The commands below reproduce the Standard SB3 2.8.0 navigation comparison on a
new computer. PPO/DDPG use `PersistentNavigationEnv`; they do not use the finite
energy experiment environment.

## 1. Install uv and clone

```bash
uv --version
git clone https://github.com/ZhangJL16/EA-MAPPO.git
cd EA-MAPPO
git checkout master
git pull --ff-only github master || git pull --ff-only origin master
cd review_bundle
```

Record the exact code before training:

```bash
git rev-parse HEAD
```

## 2. Create the repository-local environment

The solved machine used Python 3.12.3. Select one backend:

```bash
PYTHON_VERSION=3.12 TORCH_BACKEND=auto bash scripts/setup_uv_env.sh
# or TORCH_BACKEND=cpu
# or TORCH_BACKEND=cu128
```

The script creates `review_bundle/.venv`, pins NumPy 1.26.4, Gymnasium 1.2.3,
Stable-Baselines3 2.8.0 and Torch 2.7.1, prints CUDA/GPU information, and runs an
environment import/step smoke test.

## 3. Unit tests

```bash
PYTHONPATH=. .venv/bin/python -m unittest -v \
  tests.test_sb3_persistent_navigation \
  tests.test_sb3_sac_training_protocol \
  tests.test_sb3_algorithm_baselines
```

## 4. PPO and DDPG smoke tests

```bash
PYTHONPATH=. .venv/bin/python scripts/train_sb3_ppo_navigation.py \
  --seed 0 --steps 2048 --checkpoint-steps 2048 \
  --heldout-seeds 100 --evaluation-steps 100 --device auto \
  --output-dir artifacts/smoke_sb3_ppo/seed0

PYTHONPATH=. .venv/bin/python scripts/train_sb3_ddpg_navigation.py \
  --seed 0 --steps 1000 --checkpoint-steps 1000 \
  --heldout-seeds 100 --evaluation-steps 100 --device auto \
  --output-dir artifacts/smoke_sb3_ddpg/seed0
```

Smoke tests only verify API, finite losses, checkpoint save/load and evaluation.
They are not performance gates.

## 5. Inspect hardware before formal runs

```bash
nproc
nvidia-smi || true
.venv/bin/python - <<'PY'
import torch
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    print(index, torch.cuda.get_device_name(index), properties.total_memory / 1024**3, "GiB")
PY
```

## 6. Launch formal 1M runs

One GPU defaults to a serial seed queue:

```bash
PYTHON_BIN="$PWD/.venv/bin/python" DEVICE=cuda:0 RUN_MODE=auto \
  bash scripts/launch_sb3_ppo_1m.sh

# After PPO completes:
PYTHON_BIN="$PWD/.venv/bin/python" DEVICE=cuda:0 RUN_MODE=auto \
  bash scripts/launch_sb3_ddpg_1m.sh
```

To run PPO then DDPG in one conservative serial queue:

```bash
PYTHON_BIN="$PWD/.venv/bin/python" DEVICE=cuda:0 COMPARISON_MODE=serial \
  bash scripts/launch_sb3_ppo_ddpg_1m.sh
```

Do not select parallel mode on one GPU without checking memory and throughput.
The launcher rejects shared-GPU parallel mode unless `ALLOW_SHARED_GPU=1` is
set explicitly. With three or more GPUs, parallel seeds map to physical GPUs
0, 1 and 2. SB3 also warns that MLP PPO often trains faster on CPU than GPU;
`DEVICE=cpu` is therefore a valid compute choice and does not change the PPO
algorithm or environment protocol.
PPO records requested and rollout-aligned actual steps. DDPG uses
`NormalActionNoise(mean=[0,0,0], sigma=[0.1,0.1,0.1])` during training.

## 7. Monitor and verify completion

```bash
tmux list-sessions
tail -f artifacts/phase1_sb3_ppo_1m/seed0/train.log
tail -f artifacts/phase1_sb3_ddpg_1m/seed0/train.log
find artifacts/phase1_sb3_ppo_1m artifacts/phase1_sb3_ddpg_1m \
  -name COMPLETED.json -o -name FAILED.json
```

A run is complete only when `COMPLETED.json` exists and `exit_code.txt` contains
`0`. Preserve `FAILED.json` and logs if a run fails.

## 8. Summarize results

```bash
.venv/bin/python scripts/summarize_sb3_navigation_baselines.py
```

Primary DDPG comparison uses deterministic actor evaluation. Files labeled
`ddpg_exploration_noise` are exploration-noise evaluations, not stochastic
policy results. Do not combine Track B conclusions with Track A charging claims.
