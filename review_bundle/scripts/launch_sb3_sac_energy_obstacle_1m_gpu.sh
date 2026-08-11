#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$root"
echo "PREPARED_ONLY: this launcher must not be used before energy-open results are reviewed." >&2
echo "To launch deliberately: ALLOW_ENERGY_OBSTACLE=1 bash $0" >&2
[[ "${ALLOW_ENERGY_OBSTACLE:-0}" == "1" ]] || exit 2
gpu="${PHYSICAL_GPU_INDEX:-0}"
for seed in 0 1 2; do
  session="sb3-sac-energy-obstacle1m-s${seed}"; output="artifacts/phase2_sb3_sac_energy_obstacle_1m/seed${seed}"
  tmux new-session -d -s "$session" -c "$root" "bash scripts/run_one_sb3_sac_energy_obstacle_1m_gpu.sh $seed $output $gpu"
  echo "launched seed=$seed session=$session gpu=$gpu output=$output"
done
