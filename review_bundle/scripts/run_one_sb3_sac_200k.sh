#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SEED OUTPUT_DIR" >&2
  exit 2
fi

seed="$1"
output_dir="$2"
python_bin="${PYTHON_BIN:-/home/zjl/mappo/.venv/bin/python}"

mkdir -p "$output_dir"
if [[ -e "$output_dir/config.json" ]]; then
  echo "refusing to overwrite existing run: $output_dir/config.json" >&2
  exit 3
fi

command=(
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  "$python_bin" -u scripts/train_sb3_persistent_sac.py
  --scenario random_persistent_open.json
  --seed "$seed"
  --steps 200000
  --max-episode-steps 5000
  --navigation-energy-capacity 1000.0
  --goal-radius 0.20
  --minimum-goal-separation 0.60
  --sampling-margin 0.20
  --progress-weight 2.5
  --velocity-reward-weight 0.1
  --time-cost 0.01
  --completion-reward 10.0
  --collision-penalty 1.2
  --energy-cost-weight 0.01
  --backup-intervention-cost 0.1
  --learning-rate 0.0003
  --buffer-size 1000000
  --learning-starts 5000
  --batch-size 256
  --tau 0.005
  --gamma 0.99
  --train-frequency 1
  --gradient-steps 1
  --checkpoint-steps 10000 50000 100000 200000
  --heldout-seeds 100 101 102 103 104
  --evaluation-steps 2000
  --log-interval 1000
  --torch-threads 1
  --device cpu
  --output-dir "$output_dir"
)

printf '%q' "${command[0]}" > "$output_dir/command.txt"
for argument in "${command[@]:1}"; do
  printf ' %q' "$argument" >> "$output_dir/command.txt"
done
printf '\n' >> "$output_dir/command.txt"
set +e
"${command[@]}" >> "$output_dir/train.log" 2>&1 &
python_pid=$!
printf '%s\n' "$python_pid" > "$output_dir/pid.txt"
wait "$python_pid"
status=$?
set -e
printf '%s\n' "$status" > "$output_dir/exit_code.txt"
exit "$status"
