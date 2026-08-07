from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .metrics import write_csv


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict, key: str) -> float | None:
    value = row.get(key)
    if value in (None, "", "None"):
        return None
    return float(value)


def _curve_rows(comparison_root: Path, metric: str, bin_size: int = 1000) -> list[dict]:
    groups: dict[tuple[str, str, int], dict[int, list[float]]] = {}
    for path in comparison_root.glob("mission_*/*/seed_*/episode_metrics.csv"):
        scenario, method = path.parts[-4], path.parts[-3]
        seed = int(path.parts[-2].split("_")[-1])
        end_step = 0
        for row in _csv(path):
            end_step += int(float(row["episode_length"]))
            value = _float(row, metric)
            if value is None:
                continue
            bin_end = bin_size * max(1, int(np.ceil(end_step / bin_size)))
            groups.setdefault((method, scenario, bin_end), {}).setdefault(seed, []).append(value)
    output = []
    for (method, scenario, bin_end), seed_samples in sorted(groups.items()):
        seed_means = np.asarray([np.mean(values) for values in seed_samples.values()], dtype=np.float64)
        std = float(np.std(seed_means))
        half = 1.96 * std / np.sqrt(max(1, len(seed_means)))
        output.append({
            "method": method,
            "scenario": scenario,
            "environment_step": bin_end,
            "mean": float(np.mean(seed_means)),
            "std": std,
            "ci95_low": float(np.mean(seed_means) - half),
            "ci95_high": float(np.mean(seed_means) + half),
            "seeds": len(seed_means),
        })
    return output


def generate_paper_tables(
    comparison_root: str | Path = "artifacts/comparison",
    paper_root: str | Path = "artifacts/paper",
) -> None:
    comparison = Path(comparison_root)
    paper = Path(paper_root)
    paper.mkdir(parents=True, exist_ok=True)
    summary = json.loads((comparison / "aggregate/summary.json").read_text(encoding="utf-8"))
    main = []
    intervention = []
    energy = []
    runtime = []
    for row in summary:
        main.append({
            "method": row["method"], "scenario": row["scenario"],
            "task_success_mean": row.get("task_success_rate_mean"), "task_success_std": row.get("task_success_rate_std"),
            "return_success_mean": row.get("return_success_rate_mean"), "return_success_std": row.get("return_success_rate_std"),
            "collision_mean": row.get("collision_episode_rate_mean"), "collision_std": row.get("collision_episode_rate_std"),
            "episode_return_mean": row.get("mean_episode_return_mean"), "episode_return_std": row.get("mean_episode_return_std"),
            "outbound_intervention_mean": row.get("outbound_fallback_rate_mean"),
            "fallback_mean": row.get("fallback_rate_mean"),
            "zonotope_volume_mean": row.get("mean_zonotope_volume_mean"),
        })
        intervention.append({
            "method": row["method"], "scenario": row["scenario"],
            "outbound_intervention_mean": row.get("outbound_fallback_rate_mean"),
            "overall_fallback_mean": row.get("fallback_rate_mean"),
            "no_generator_mean": row.get("no_generator_set_rate_mean"),
            "certificate_failure_mean": row.get("certificate_failure_rate_mean"),
        })
        energy.append({
            "method": row["method"], "scenario": row["scenario"],
            "return_success_mean": row.get("return_success_rate_mean"),
            "energy_depletion_mean": row.get("energy_depletion_rate_mean"),
            "terminal_energy_mean": row.get("terminal_energy_mean_mean"),
            "minimum_energy_margin_mean": row.get("minimum_energy_margin_mean"),
        })
        runtime.append({
            "method": row["method"], "scenario": row["scenario"],
            "T_policy_p99": row.get("T_policy_p99_seconds_mean"),
            "T_certificate_p99": row.get("T_certificate_p99_seconds_mean"),
            "T_watchdog_p99": row.get("T_watchdog_p99_seconds_mean"),
            "T_plant_p99": row.get("T_plant_p99_seconds_mean"),
            "T_total_p99": row.get("T_total_p99_seconds_mean"),
            "timing_scope": "desktop profiling, not WCET",
        })
    write_csv(paper / "main_comparison.csv", main)
    write_csv(paper / "intervention_breakdown.csv", intervention)
    write_csv(paper / "energy_return.csv", energy)
    write_csv(paper / "runtime.csv", runtime)
    curve_mapping = {
        "learning_curve_task_success.csv": "task_success",
        "learning_curve_return_success.csv": "return_success",
        "learning_curve_episode_return.csv": "episode_return",
        "learning_curve_intervention.csv": "outbound_fallback_rate",
    }
    for filename, metric in curve_mapping.items():
        write_csv(paper / filename, _curve_rows(comparison, metric))

