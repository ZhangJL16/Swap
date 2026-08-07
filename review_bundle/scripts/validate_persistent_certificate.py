#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.certified_uav import make_persistent_uav_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Build synthetic persistent service manifests and report their gates.")
    parser.add_argument("--scenarios", nargs="+", default=["persistent_open", "persistent_obstacle", "persistent_energy_tight"])
    parser.add_argument("--output", default="artifacts/persistent/certificate_gate.json")
    args = parser.parse_args()
    results = []
    for name in args.scenarios:
        env = make_persistent_uav_env(f"{name}.json", timing_mode="functional")
        _, info = env.reset(seed=0)
        manifest = env.certificate_provider.persistent_manifest
        results.append({
            "scenario": name,
            "gate": "PASS" if manifest.gate_pass else "FAIL",
            "manifest_hash": manifest.manifest_hash,
            "network_hash": manifest.service_network_hash,
            "edge_count": len(manifest.edge_certificates),
            "failure_reasons": manifest.failure_reasons,
            "synthetic_only": True,
        })
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    if not all(item["gate"] == "PASS" for item in results):
        raise SystemExit("PERSISTENT_CERTIFICATE_GATE = FAIL")
    print("PERSISTENT_CERTIFICATE_GATE = PASS")


if __name__ == "__main__":
    main()
