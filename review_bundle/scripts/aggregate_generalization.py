from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.metrics import write_csv


def main() -> None:
    paper = Path("artifacts/paper")
    rows = []
    for path in sorted(paper.glob("generalization_*.csv")):
        if path.name == "generalization.csv":
            continue
        with path.open(encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    write_csv(paper / "generalization.csv", rows)
    summary_rows = []
    metrics = (
        "task_success", "return_success", "collision", "outbound_intervention",
        "path_length", "energy_consumed",
    )
    for family in sorted({row["scenario_family"] for row in rows}):
        for method in sorted({row["method"] for row in rows}):
            group = [row for row in rows if row["scenario_family"] == family and row["method"] == method]
            if not group:
                continue
            summary = {
                "scenario_family": family,
                "method": method,
                "heldout_scenarios": len(group),
                "checkpoint_seeds": len({row["seed"] for row in group}),
                "evaluation_episodes_per_scenario": int(float(group[0]["episodes"])),
                "scope": "single-checkpoint-seed held-out synthetic empirical evaluation",
            }
            for metric in metrics:
                values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
                mean = float(values.mean())
                std = float(values.std())
                half_width = 1.96 * std / np.sqrt(len(values))
                summary[f"{metric}_mean"] = mean
                summary[f"{metric}_std"] = std
                summary[f"{metric}_ci95_low"] = mean - half_width
                summary[f"{metric}_ci95_high"] = mean + half_width
            summary_rows.append(summary)
    write_csv(paper / "generalization_summary.csv", summary_rows)
    scenarios = {row["scenario_id"] for row in rows}
    methods = {row["method"] for row in rows}
    violations = [
        row for row in rows
        if float(row.get("collision", 0.0)) > 0.0 and row["method"] in {"generator_sac", "center_only", "shield_sac"}
    ]
    uncertified = sum(int(float(row.get("uncertified_task_publication_count", 0))) for row in rows)
    invalid_kappa = sum(int(float(row.get("invalid_kappa_fallback_count", 0))) for row in rows)
    gate = {
        "GENERALIZATION_GATE": "PASS" if len(scenarios) == 20 and {"generator_sac", "center_only", "shield_sac"} <= methods and not violations and uncertified == 0 and invalid_kappa == 0 else "BLOCKED",
        "heldout_scenarios": len(scenarios),
        "methods": sorted(methods),
        "rows": len(rows),
        "sampled_certified_method_collisions": len(violations),
        "uncertified_task_publication_count": uncertified,
        "invalid_kappa_fallback_count": invalid_kappa,
        "scope": "synthetic held-out generalization evidence only",
    }
    (paper / "generalization_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
