#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SEED OUTPUT_DIR" >&2
  exit 2
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
seed="$1"
output_dir="$2"
python_bin="${PYTHON_BIN:-$root/.venv/bin/python}"
device="${DEVICE:-auto}"
mkdir -p "$output_dir"
if [[ -e "$output_dir/config.json" ]]; then
  echo "refusing to overwrite existing run: $output_dir/config.json" >&2
  exit 3
fi
command=("$python_bin" -u scripts/train_sb3_ppo_navigation.py --seed "$seed" --steps 1000000 --device "$device" --output-dir "$output_dir")
printf '%q ' "${command[@]}" > "$output_dir/command.txt"
printf '\n' >> "$output_dir/command.txt"
set +e
"${command[@]}" >> "$output_dir/train.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$output_dir/pid.txt"
wait "$pid"
status=$?
set -e
printf '%s\n' "$status" > "$output_dir/exit_code.txt"
exit "$status"
