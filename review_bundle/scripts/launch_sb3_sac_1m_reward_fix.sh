#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

for seed in 0 1 2; do
  session="sb3-sac1m-rfix-s${seed}"
  output_dir="artifacts/phase1_sb3_sac_1m_reward_fix/seed${seed}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "refusing to replace existing tmux session: $session" >&2
    exit 4
  fi
  if [[ -e "$output_dir/config.json" ]]; then
    echo "refusing to overwrite existing run: $output_dir/config.json" >&2
    exit 5
  fi
done

for seed in 0 1 2; do
  session="sb3-sac1m-rfix-s${seed}"
  output_dir="artifacts/phase1_sb3_sac_1m_reward_fix/seed${seed}"
  tmux new-session -d -s "$session" -c "$root" \
    "bash scripts/run_one_sb3_sac_1m_reward_fix.sh $seed $output_dir"
  echo "launched seed=$seed session=$session output=$output_dir"
done
