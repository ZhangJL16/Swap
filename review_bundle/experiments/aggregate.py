from __future__ import annotations

import json
from pathlib import Path
import csv
import numpy as np


def aggregate_results(root: str | Path) -> list[dict]:
    root = Path(root)
    records = [json.loads(path.read_text(encoding="utf-8")) for path in root.glob("mission_*/*/seed_*/runtime_profile.json")]
    grouped: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        grouped.setdefault((record["method"], record["scenario"]), []).append(record)
    rows = []
    metrics = ("task_success_rate", "return_success_rate", "collision_episode_rate", "energy_depletion_rate", "mean_episode_return", "fallback_rate", "mean_zonotope_volume", "runtime_p99_seconds", "wall_time_seconds")
    for (method, scenario), values in sorted(grouped.items()):
        row = {"method": method, "scenario": scenario, "seeds": len(values)}
        for metric in metrics:
            samples = [float(value[metric]) for value in values]
            row[f"{metric}_mean"] = float(np.mean(samples))
            row[f"{metric}_std"] = float(np.std(samples))
        rows.append(row)
    aggregate = root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    (aggregate / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        with (aggregate / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    safety_fields = [field for field in rows[0] if field in {"method", "scenario", "seeds"} or any(token in field for token in ("task_success", "return_success", "collision", "energy_depletion", "fallback", "zonotope"))] if rows else []
    runtime_fields = [field for field in rows[0] if field in {"method", "scenario", "seeds"} or "wall_time" in field] if rows else []
    for path, fields in ((aggregate / "safety_performance_table.csv", safety_fields), (aggregate / "runtime_table.csv", runtime_fields)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            if fields:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader(); writer.writerows({field: row[field] for field in fields} for row in rows)
    curve_groups: dict[tuple[str, str, int], list[float]] = {}
    for path in root.glob("mission_*/*/seed_*/training_metrics.csv"):
        scenario, method = path.parts[-4], path.parts[-3]
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                step = int(row["environment_step"])
                bin_step = 100 * ((step - 1) // 100 + 1)
                curve_groups.setdefault((method, scenario, bin_step), []).append(float(row["reward"]))
    curve_rows = [
        {"method": method, "scenario": scenario, "environment_step": step, "reward_mean": float(np.mean(values)), "reward_std": float(np.std(values)), "samples": len(values)}
        for (method, scenario, step), values in sorted(curve_groups.items())
    ]
    with (aggregate / "learning_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        if curve_rows:
            writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
            writer.writeheader(); writer.writerows(curve_rows)
    return rows
