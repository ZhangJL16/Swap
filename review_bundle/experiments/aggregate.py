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
    metrics = (
        "task_success_rate", "return_success_rate", "collision_episode_rate", "energy_depletion_rate",
        "evaluation_task_success_rate", "evaluation_return_success_rate",
        "mean_episode_return", "fallback_rate", "outbound_fallback_rate", "no_generator_set_rate",
        "mean_zonotope_volume", "min_zonotope_volume", "mean_sigma_min", "mean_condition_number",
        "terminal_energy_mean", "minimum_energy_margin", "minimum_distance_to_task",
        "uncertified_task_publication_count", "fallback_with_invalid_kappa",
        "premature_terminal_rate", "corridor_exit_rate", "certificate_failure_rate", "velocity_failure_rate", "other_failure_rate",
        "runtime_p99_seconds", "T_policy_p99_seconds", "T_certificate_p99_seconds", "T_watchdog_p99_seconds", "T_plant_p99_seconds", "T_total_p99_seconds",
        "wall_time_seconds",
    )
    for (method, scenario), values in sorted(grouped.items()):
        row = {"method": method, "scenario": scenario, "seeds": len(values)}
        for metric in metrics:
            samples = [float(value[metric]) for value in values if value.get(metric) is not None]
            row[f"{metric}_mean"] = float(np.mean(samples)) if samples else None
            row[f"{metric}_std"] = float(np.std(samples)) if samples else None
        rows.append(row)
    aggregate = root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    (aggregate / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        with (aggregate / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
    safety_fields = [field for field in rows[0] if field in {"method", "scenario", "seeds"} or any(token in field for token in ("task_success", "return_success", "collision", "energy_depletion", "fallback", "zonotope"))] if rows else []
    runtime_fields = [field for field in rows[0] if field in {"method", "scenario", "seeds"} or "runtime" in field or field.startswith("T_") or "wall_time" in field] if rows else []
    for path, fields in ((aggregate / "safety_performance_table.csv", safety_fields), (aggregate / "runtime_table.csv", runtime_fields)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            if fields:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                writer.writeheader(); writer.writerows({field: row[field] for field in fields} for row in rows)
    pareto_rows = [
        {
            "method": row["method"], "scenario": row["scenario"],
            "task_success": row.get("task_success_rate_mean"),
            "episode_return": row.get("mean_episode_return_mean"),
            "collision_rate": row.get("collision_episode_rate_mean"),
            "return_failure_rate": None if row.get("return_success_rate_mean") is None else 1.0 - row["return_success_rate_mean"],
            "fallback_rate": row.get("fallback_rate_mean"),
        }
        for row in rows
    ]
    with (aggregate / "safety_performance.csv").open("w", newline="", encoding="utf-8") as handle:
        if pareto_rows:
            writer = csv.DictWriter(handle, fieldnames=list(pareto_rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(pareto_rows)
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
            writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(curve_rows)
    generator_diagnostics = []
    for path in root.glob("mission_*/generator_sac/seed_*/trajectory_diagnostics.csv"):
        with path.open(encoding="utf-8") as handle:
            trajectory = list(csv.DictReader(handle))
        centers = {row.get("zonotope_center", "") for row in trajectory if row.get("zonotope_center", "")}
        generators = {row.get("zonotope_G", "") for row in trajectory if row.get("zonotope_G", "")}
        volumes = {row.get("zonotope_volume", "") for row in trajectory if row.get("zonotope_volume", "")}
        generator_diagnostics.append({
            "scenario": path.parts[-4], "seed": int(path.parts[-2].split("_")[-1]),
            "unique_c": len(centers), "unique_G": len(generators), "unique_zonotope_volume": len(volumes),
            "state_dependent": int(len(centers) > 1 and len(generators) > 1 and len(volumes) > 1),
        })
    with (aggregate / "state-dependent-generator-diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        if generator_diagnostics:
            writer = csv.DictWriter(handle, fieldnames=list(generator_diagnostics[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(generator_diagnostics)
    return rows
