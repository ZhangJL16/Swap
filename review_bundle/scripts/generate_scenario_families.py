from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.certified_uav import make_certified_uav_env
from envs.certified_uav.scenario_families import generate_scenario_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/scenario_families")
    parser.add_argument("--training", type=int, default=20)
    parser.add_argument("--validation", type=int, default=10)
    parser.add_argument("--heldout", type=int, default=20)
    parser.add_argument("--validate-certificates", action="store_true")
    args = parser.parse_args()
    records = generate_scenario_splits(
        args.output_dir,
        split_sizes={"training": args.training, "validation": args.validation, "heldout": args.heldout},
    )
    closed = []
    for record in records:
        if not args.validate_certificates:
            closed.append(record)
            continue
        runtime = make_certified_uav_env(record.path, timing_mode="functional")
        runtime.reset(seed=record.seed)
        report = runtime.mission_provider.validation_report()
        closed.append(replace(
            record,
            certificate_manifest_hash=runtime.mission_provider.manifest.manifest_hash,
            certificate_gate=report["mission_certificate_gate"],
        ))
        print(json.dumps(asdict(closed[-1]), sort_keys=True), flush=True)
    output = Path(args.output_dir) / "scenario_index.json"
    output.write_text(json.dumps([asdict(record) for record in closed], indent=2), encoding="utf-8")
    failures = [record.scenario_id for record in closed if args.validate_certificates and record.certificate_gate != "PASS"]
    print(json.dumps({"scenarios": len(closed), "gate_failures": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
