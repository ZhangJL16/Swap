#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

python_version="${PYTHON_VERSION:-3.12}"
torch_backend="${TORCH_BACKEND:-auto}"
venv_dir="${VENV_DIR:-$root/.venv}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 2
fi
uv --version

if [[ "$torch_backend" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    torch_backend="cu128"
  else
    torch_backend="cpu"
  fi
fi
if [[ "$torch_backend" != "cpu" && "$torch_backend" != "cu128" ]]; then
  echo "TORCH_BACKEND must be auto, cpu, or cu128" >&2
  exit 3
fi

if [[ -x "$venv_dir/bin/python" ]]; then
  echo "reusing existing virtual environment: $venv_dir"
else
  uv venv --python "$python_version" "$venv_dir"
fi
python_bin="$venv_dir/bin/python"
torch_index="https://download.pytorch.org/whl/$torch_backend"
uv pip install --python "$python_bin" torch==2.7.1 --index-url "$torch_index"
uv pip install --python "$python_bin" -r requirements-sb3-baselines.txt

PYTHONPATH=. "$python_bin" - <<'PY'
import importlib.metadata
import json
import numpy
import gymnasium
import torch
from envs.certified_uav import PersistentNavigationEnv

environment = PersistentNavigationEnv(max_episode_steps=2)
observation, _ = environment.reset(seed=0)
observation, reward, terminated, truncated, info = environment.step(environment.action_space.sample())
result = {
    "python": __import__("sys").version,
    "numpy": numpy.__version__,
    "gymnasium": gymnasium.__version__,
    "stable_baselines3": importlib.metadata.version("stable-baselines3"),
    "torch": torch.__version__,
    "torch_cuda_available": torch.cuda.is_available(),
    "torch_cuda_device_count": torch.cuda.device_count(),
    "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    "environment_observation_shape": list(observation.shape),
    "environment_reward_finite": bool(numpy.isfinite(reward)),
    "environment_terminated": terminated,
    "environment_truncated": truncated,
    "action_shape": list(environment.action_space.shape),
    "collision_telemetry_present": "collision" in info,
}
print(json.dumps(result, indent=2, sort_keys=True))
environment.close()
PY
