#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 SEED OUTPUT_DIR PHYSICAL_GPU_INDEX" >&2
  exit 2
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
seed="$1"; output_dir="$2"; gpu="$3"
python_bin="${PYTHON_BIN:-$root/.venv/bin/python}"
mkdir -p "$output_dir"
[[ ! -e "$output_dir/config.json" ]] || { echo "refusing to overwrite $output_dir" >&2; exit 3; }
command=(env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$python_bin" -u scripts/train_sb3_energy_sac.py --scenario random_persistent_open.json --seed "$seed" --steps 1000000 --device cuda:0 --output-dir "$output_dir")
printf '%q ' "${command[@]}" > "$output_dir/command.txt"; printf '\n' >> "$output_dir/command.txt"
set +e
"${command[@]}" >> "$output_dir/train.log" 2>&1 &
pid=$!; printf '%s\n' "$pid" > "$output_dir/pid.txt"; wait "$pid"; status=$?
set -e
printf '%s\n' "$status" > "$output_dir/exit_code.txt"; exit "$status"
