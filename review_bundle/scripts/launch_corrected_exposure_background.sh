#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=/home/zjl/mappo/.venv/bin/python
OUTPUT_ROOT="$ROOT/artifacts/corrected_exposure_5k"
mkdir -p "$OUTPUT_ROOT" "$ROOT/artifacts/random_persistent"

pids=()
for seed in 0 1 2; do
  seed_dir="$OUTPUT_ROOT/seed${seed}"
  mkdir -p "$seed_dir"
  log="$seed_dir/pipeline.log"
  session="corrected_exposure_seed${seed}"
  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" -c "$ROOT" \
    "exec '$PYTHON' '$ROOT/scripts/run_corrected_exposure_pipeline.py' --seed '$seed' --output-root artifacts/corrected_exposure_5k >'$log' 2>&1"
  pids+=("$(tmux list-panes -t "$session" -F '#{pane_pid}')")
done

finalizer_session="corrected_exposure_finalizer"
tmux kill-session -t "$finalizer_session" 2>/dev/null || true
tmux new-session -d -s "$finalizer_session" -c "$ROOT" \
  "exec '$PYTHON' '$ROOT/scripts/finalize_corrected_exposure_pipeline.py' >'$OUTPUT_ROOT/finalizer.log' 2>&1"
finalizer_pid="$(tmux list-panes -t "$finalizer_session" -F '#{pane_pid}')"

"$PYTHON" - "${pids[@]}" "$finalizer_pid" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

root = Path.cwd()
pids = [int(value) for value in sys.argv[1:4]]
finalizer_pid = int(sys.argv[4])
payload = {
    "launched_at": datetime.now(timezone.utc).isoformat(),
    "worker_pids": {f"seed{seed}": pid for seed, pid in enumerate(pids)},
    "finalizer_pid": finalizer_pid,
    "worker_logs": {f"seed{seed}": f"artifacts/corrected_exposure_5k/seed{seed}/pipeline.log" for seed in range(3)},
}
(root / "artifacts/corrected_exposure_5k/launch_manifest.json").write_text(
    json.dumps(payload, indent=2), encoding="utf-8"
)
print(json.dumps(payload))
PY
