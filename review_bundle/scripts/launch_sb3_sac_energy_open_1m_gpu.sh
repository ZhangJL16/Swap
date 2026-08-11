#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$root"
python_bin="${PYTHON_BIN:-$root/.venv/bin/python}"; gpu="${PHYSICAL_GPU_INDEX:-0}"
gpu_count="$($python_bin -c 'import torch; print(torch.cuda.device_count())')"
[[ "$gpu_count" -gt "$gpu" ]] || { echo "requested GPU $gpu unavailable" >&2; exit 3; }
for seed in 0 1 2; do
  session="sb3-sac-energy-open1m-s${seed}"; output="artifacts/phase2_sb3_sac_energy_open_1m/seed${seed}"
  [[ ! -e "$output/config.json" ]] || { echo "existing energy run seed${seed}" >&2; exit 4; }
  tmux new-session -d -s "$session" -c "$root" "bash scripts/run_one_sb3_sac_energy_open_1m_gpu.sh $seed $output $gpu"
  echo "launched seed=$seed session=$session gpu=$gpu output=$output"
done
