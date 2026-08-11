#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
python_bin="${PYTHON_BIN:-$root/.venv/bin/python}"
gpu_count="$($python_bin -c 'import torch; print(torch.cuda.device_count())')"
cpu_cores="$(getconf _NPROCESSORS_ONLN)"
mode="${COMPARISON_MODE:-serial}"
echo "cpu_cores=$cpu_cores gpu_count=$gpu_count comparison_mode=$mode"
$python_bin - <<'PY'
import json, torch
print(json.dumps({
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count(),
    "gpus": [{"index": i, "name": torch.cuda.get_device_name(i), "vram_gib": torch.cuda.get_device_properties(i).total_memory / 1024**3} for i in range(torch.cuda.device_count())],
}, indent=2))
PY
if [[ "$mode" == "serial" ]]; then
  session="sb3-ppo-ddpg1m-queue"
  command="set -euo pipefail; export CUDA_VISIBLE_DEVICES=${PHYSICAL_GPU_INDEX:-0} DEVICE=${DEVICE:-auto};"
  for algorithm in ppo ddpg; do
    for seed in 0 1 2; do
      command+=" bash scripts/run_one_sb3_${algorithm}_1m.sh $seed artifacts/phase1_sb3_${algorithm}_1m/seed$seed;"
    done
  done
  tmux new-session -d -s "$session" -c "$root" "$command"
  echo "launched serial PPO then DDPG queue session=$session"
elif [[ "$mode" == "parallel_algorithms" && "$gpu_count" -ge 2 ]]; then
  PHYSICAL_GPU_INDEX=0 RUN_MODE=serial DEVICE=cuda:0 bash scripts/launch_sb3_ppo_1m.sh
  PHYSICAL_GPU_INDEX=1 RUN_MODE=serial DEVICE=cuda:0 bash scripts/launch_sb3_ddpg_1m.sh
else
  echo "Use COMPARISON_MODE=serial, or parallel_algorithms with at least two GPUs" >&2
  exit 4
fi
