from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .metrics import write_csv


ABLATION_METRICS = (
    "task_success",
    "return_success",
    "collision",
    "mission_completion_steps",
    "task_completion_steps",
    "total_path_length",
    "total_energy_consumed",
    "episode_return",
    "terminal_energy",
    "outbound_intervention_rate",
    "return_handoff_rate",
    "mean_residual_norm",
    "mean_residual_to_center_ratio",
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(value: str | None) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def summarize_evaluations(root: str | Path, output: str | Path) -> list[dict]:
    root = Path(root)
    grouped: dict[tuple[str, str], list[tuple[int, dict[str, float]]]] = {}
    for path in root.glob("mission_*/*/seed_*/evaluation_metrics.csv"):
        rows = _read_rows(path)
        if not rows:
            continue
        scenario, method = path.parts[-4], path.parts[-3]
        seed = int(path.parts[-2].split("_")[-1])
        seed_metrics = {}
        for metric in ABLATION_METRICS:
            samples = [_number(row.get(metric)) for row in rows]
            finite = [value for value in samples if value is not None and np.isfinite(value)]
            seed_metrics[metric] = float(np.mean(finite)) if finite else float("nan")
        grouped.setdefault((scenario, method), []).append((seed, seed_metrics))
    summaries: list[dict] = []
    for (scenario, method), seed_values in sorted(grouped.items()):
        row = {"scenario": scenario, "method": method, "seeds": len(seed_values)}
        for metric in ABLATION_METRICS:
            samples = np.asarray([values[metric] for _, values in seed_values], dtype=np.float64)
            samples = samples[np.isfinite(samples)]
            if samples.size:
                row[f"{metric}_mean"] = float(np.mean(samples))
                row[f"{metric}_std"] = float(np.std(samples))
                row[f"{metric}_median"] = float(np.median(samples))
            else:
                row[f"{metric}_mean"] = row[f"{metric}_std"] = row[f"{metric}_median"] = None
        summaries.append(row)
    write_csv(Path(output), summaries)
    return summaries


def paired_bootstrap(
    root: str | Path,
    scenario: str,
    metric: str,
    left_method: str = "center_only",
    right_method: str = "generator_sac",
    *,
    draws: int = 20_000,
    seed: int = 20260807,
) -> dict:
    root = Path(root)
    per_method: dict[str, dict[int, float]] = {}
    for method in (left_method, right_method):
        values = {}
        for path in (root / scenario / method).glob("seed_*/evaluation_metrics.csv"):
            samples = [
                value for value in (_number(row.get(metric)) for row in _read_rows(path))
                if value is not None and np.isfinite(value)
            ]
            if samples:
                values[int(path.parent.name.split("_")[-1])] = float(np.mean(samples))
        per_method[method] = values
    common = sorted(set(per_method[left_method]) & set(per_method[right_method]))
    if not common:
        return {"scenario": scenario, "metric": metric, "status": "missing-paired-seeds"}
    differences = np.asarray([
        per_method[right_method][seed_value] - per_method[left_method][seed_value]
        for seed_value in common
    ])
    rng = np.random.default_rng(seed)
    boot = differences[rng.integers(0, len(differences), size=(draws, len(differences)))].mean(axis=1)
    return {
        "scenario": scenario,
        "metric": metric,
        "left_method": left_method,
        "right_method": right_method,
        "paired_seeds": len(common),
        "mean_difference_right_minus_left": float(differences.mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "interpretation": "synthetic empirical paired-bootstrap interval; not a safety proof",
    }
