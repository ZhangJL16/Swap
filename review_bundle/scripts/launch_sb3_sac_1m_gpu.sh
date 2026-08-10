#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

physical_gpu_index="${PHYSICAL_GPU_INDEX:-0}"
gpu_count="$(${PYTHON_BIN:-/home/zjl/mappo/.venv/bin/python} -c 'import torch; print(torch.cuda.device_count())')"

if [[ "$gpu_count" -lt 1 ]]; then
  echo "CUDA is unavailable; refusing to launch formal GPU runs" >&2
  exit 4
fi
if [[ "$physical_gpu_index" -ge "$gpu_count" ]]; then
  echo "physical GPU index $physical_gpu_index is outside detected count $gpu_count" >&2
  exit 5
fi
for seed in 0 1 2; do
  session="sb3-sac1m-gpu-s${seed}"
  output_dir="artifacts/phase1_sb3_sac_1m_gpu/seed${seed}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "refusing to replace existing tmux session: $session" >&2
    exit 6
  fi
  if [[ -e "$output_dir/config.json" ]]; then
    echo "refusing to overwrite existing run: $output_dir/config.json" >&2
    exit 7
  fi
done

for seed in 0 1 2; do
  session="sb3-sac1m-gpu-s${seed}"
  output_dir="artifacts/phase1_sb3_sac_1m_gpu/seed${seed}"
  tmux new-session -d -s "$session" -c "$root" \
    "bash scripts/run_one_sb3_sac_1m_gpu.sh $seed $output_dir $physical_gpu_index"
  echo "launched seed=$seed session=$session physical_gpu_index=$physical_gpu_index output=$output_dir"
done
