#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
python_bin="${PYTHON_BIN:-$root/.venv/bin/python}"
run_mode="${RUN_MODE:-auto}"
physical_gpu_index="${PHYSICAL_GPU_INDEX:-0}"
gpu_count="$($python_bin -c 'import torch; print(torch.cuda.device_count())')"
cpu_cores="$(getconf _NPROCESSORS_ONLN)"
echo "cpu_cores=$cpu_cores gpu_count=$gpu_count run_mode=$run_mode"
$python_bin - <<'PY'
import json, torch
print(json.dumps({
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count(),
    "gpus": [{"index": i, "name": torch.cuda.get_device_name(i), "vram_gib": torch.cuda.get_device_properties(i).total_memory / 1024**3} for i in range(torch.cuda.device_count())],
}, indent=2))
PY
if [[ "$run_mode" == "auto" ]]; then
  [[ "$gpu_count" -ge 3 ]] && run_mode="parallel" || run_mode="serial"
fi
for seed in 0 1 2; do
  [[ ! -e "artifacts/phase1_sb3_ppo_1m/seed${seed}/config.json" ]] || { echo "existing PPO run seed${seed}" >&2; exit 3; }
done
if [[ "$run_mode" == "serial" ]]; then
  session="sb3-ppo1m-queue"
  tmux new-session -d -s "$session" -c "$root" \
    "export CUDA_VISIBLE_DEVICES=$physical_gpu_index DEVICE=${DEVICE:-auto}; for seed in 0 1 2; do bash scripts/run_one_sb3_ppo_1m.sh \$seed artifacts/phase1_sb3_ppo_1m/seed\$seed || exit \$?; done"
  echo "launched serial PPO queue session=$session"
elif [[ "$run_mode" == "parallel" ]]; then
  if [[ "$gpu_count" -lt 3 && "${ALLOW_SHARED_GPU:-0}" != "1" ]]; then
    echo "parallel mode needs three GPUs; set ALLOW_SHARED_GPU=1 only after checking memory" >&2
    exit 5
  fi
  for seed in 0 1 2; do
    if [[ "$gpu_count" -ge 3 ]]; then gpu="$seed"; else gpu="$physical_gpu_index"; fi
    session="sb3-ppo1m-s${seed}"
    tmux new-session -d -s "$session" -c "$root" "export CUDA_VISIBLE_DEVICES=$gpu DEVICE=${DEVICE:-auto}; bash scripts/run_one_sb3_ppo_1m.sh $seed artifacts/phase1_sb3_ppo_1m/seed$seed"
    echo "launched PPO seed=$seed session=$session physical_gpu=$gpu"
  done
else
  echo "RUN_MODE must be auto, serial, or parallel" >&2
  exit 4
fi
