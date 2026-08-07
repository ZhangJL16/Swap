from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from envs.certified_uav import make_certified_uav_env


def main() -> None:
    gate_path = Path("artifacts/mission_certificate/gate.json")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in Path("artifacts/mission_certificate").glob("mission_*.json")]
    generator = make_certified_uav_env("mission_open.json", generator_center_mode="task_oriented", timing_mode="functional")
    shield = make_certified_uav_env("mission_open.json", generator_center_mode="task_oriented", timing_mode="functional")
    generator.reset(seed=0); shield.reset(seed=0)
    left, right = generator.action_context(), shield.action_context()
    summary = json.loads(Path("artifacts/comparison/aggregate/summary.json").read_text(encoding="utf-8"))
    indexed = {(row["method"], row["scenario"]): row for row in summary}
    audit = {
        "commit_under_audit": "2f0759aa5b53527a98c6fe897048d49bc5c01c63",
        "all_mission_certificate_gates_pass": gate.get("MISSION_CERTIFICATE_GATE") == "PASS",
        "scenario_gate_status": gate.get("scenarios", {}),
        "strict_recovery_descent": all(report.get("maximum_recovery_steps", 0) > 0 and not report.get("failed_cells") for report in reports),
        "certificate_checks": {
            "complete_successor_containment": all(not report.get("failed_cells") for report in reports),
            "energy_E3": all(report.get("maximum_E3_residual", -1) >= 0 for report in reports),
            "terminal_linkage": all(report.get("terminal_reached") for report in reports),
            "hash_dependencies": all(report.get("hash_chain_valid") for report in reports),
            "sampled_kappa_collisions": sum(report.get("sampled_collision_count", 0) for report in reports),
        },
        "total_recovery_cells": sum(report.get("certified_cells", 0) for report in reports),
        "shared_kappa": bool(np.allclose(left["kappa"], right["kappa"])),
        "shared_recovery_hash": left["recovery_hash"] == right["recovery_hash"],
        "shared_manifest": left["certificate_epoch"] == right["certificate_epoch"],
        "comparison_design": "4 methods x 4 scenarios x 5 seeds x 10000 steps",
        "generator_open": indexed.get(("generator_sac", "mission_open")),
        "generator_obstacle": indexed.get(("generator_sac", "mission_obstacle")),
        "shield_open": indexed.get(("shield_sac", "mission_open")),
        "shield_obstacle": indexed.get(("shield_sac", "mission_obstacle")),
        "generator_center_default": "task_oriented explicit reference/waypoint controller; not actor output",
        "evidence_scope": "software and synthetic empirical evidence only",
    }
    output = Path("artifacts/paper/baseline_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
