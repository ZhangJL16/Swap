#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _mean(runs: list[dict[str, object]], key: str) -> float:
    return float(np.mean([float(run[key]) for run in runs]))


def _reward(rows: list[dict[str, object]]) -> dict[str, float]:
    names = (
        "goal_progress_reward",
        "task_completion_reward",
        "elapsed_time_cost",
        "energy_cost",
        "backup_intervention_event_cost",
        "charging_dwell_cost",
    )
    result = {name: 0.0 for name in names}
    for row in rows:
        components = row.get("reward_components")
        if not isinstance(components, dict):
            continue
        for name in names:
            result[name] += float(components.get(name, 0.0))
    return result


def _run(directory: Path) -> dict[str, object]:
    summary = _json(directory / "summary.json")
    audit = _json(directory / "optimization_audit.json")
    rows = _jsonl(directory / "trajectory_events.jsonl")
    aggregate = summary["aggregate_metrics"]
    first = rows[0]
    initial_distance = float(np.linalg.norm(np.asarray(first["goal"]) - np.asarray(first["position_before"])))
    distances = [float(np.linalg.norm(np.asarray(row["goal"]) - np.asarray(row["position"]))) for row in rows if row.get("goal") is not None]
    early = [row for row in rows if int(row["step"]) <= 500]
    late = [row for row in rows if 1500 < int(row["step"]) <= 2000]

    def interval(selected: list[dict[str, object]], audit_name: str) -> dict[str, object]:
        temporal = audit["TEMPORAL"][audit_name]
        return {
            "goal_progress": float(sum(float(row.get("goal_progress", 0.0)) for row in selected)),
            "tasks_completed": int(sum(bool(row.get("task_completed_now", False)) for row in selected)),
            "actor_goal_projection": temporal["actor_goal_projection"]["mean"],
            "oracle_gap": temporal["oracle_gap"]["mean"],
        }

    entropy = audit["ENTROPY"]
    actor = audit["ACTOR"]
    critic = audit["CRITIC"]
    sensitivity = audit["TRAINED_GOAL_SENSITIVITY_DIAGNOSTIC"]
    safety = {
        "invalid_kappa": int(aggregate["invalid_kappa_fallback_count"]),
        "fail_closed": int(aggregate["fail_closed_steps"]),
        "collision": int(aggregate["collision_count"]),
        "depletion": int(aggregate["energy_depletion_count"]),
        "uncertified_publication": int(aggregate["uncertified_publication_count"]),
        "accepted_into_kappa_only": int(aggregate["accepted_into_kappa_only_count"]),
    }
    return {
        "variant": summary["temperature_coordinate"],
        "seed": int(summary["seed"]),
        "steps": int(summary["steps"]),
        "tasks_completed": int(aggregate["tasks_completed"]),
        "tasks_per_1000_steps": float(aggregate["tasks_per_1000_steps"]),
        "net_goal_progress": float(aggregate["total_goal_progress"]),
        "fraction_steps_progressing_to_goal": float(aggregate["fraction_steps_progressing_to_goal"]),
        "initial_distance": initial_distance,
        "minimum_distance": min(distances),
        "rl_generator_fraction": float(aggregate["rl_generator_fraction"]),
        "kappa_backup_fraction": float(aggregate["kappa_backup_fraction"]),
        "backup_intervention_events": int(aggregate["backup_intervention_reward_events"]),
        "kappa_backup_steps": int(aggregate["kappa_backup_steps"]),
        "actor_goal_projection": float(actor["actor_goal_projection"]["mean"]),
        "oracle_goal_projection": float(actor["oracle_goal_projection"]["mean"]),
        "oracle_gap": float(actor["oracle_gap"]["mean"]),
        "goal_sensitivity_latent_distance": float(sensitivity["pairwise_latent_distance"]["mean"]),
        "goal_sensitivity_action_distance": float(sensitivity["pairwise_action_distance"]["mean"]),
        "alpha": float(entropy["alpha"]),
        "mean_log_prob_u": float(entropy["mean_log_prob_u"]),
        "mean_negative_tanh_log_jacobian": float(entropy["mean_negative_tanh_log_jacobian"]),
        "mean_negative_log_det_G": float(entropy["mean_negative_log_det_G"]),
        "normalized_log_prob": float(entropy["mean_normalized_log_prob"]),
        "physical_log_prob": float(entropy["mean_physical_log_prob"]),
        "alpha_residual_used_for_training": float(entropy["alpha_residual_used_for_training"]),
        "fraction_Q_oracle_gt_Q_actor": float(critic["fraction_Q_oracle_gt_Q_actor"]),
        "fraction_Q_oracle_gt_Q_center": float(critic["fraction_Q_oracle_gt_Q_center"]),
        "fraction_Q_oracle_gt_Q_opposite": float(critic["fraction_Q_oracle_gt_Q_opposite"]),
        "mean_Q_oracle_minus_actor": float(critic["mean_Q_oracle_minus_actor"]),
        "reward_decomposition": _reward(rows),
        "early_0_500": interval(early, "early_0_500"),
        "late_1500_2000": interval(late, "late_1500_2000"),
        "safety": safety,
        "safety_pass": all(value == 0 for value in safety.values()),
        "artifact_directory": str(directory.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/random_persistent/normalized_temperature_comparison.json")
    args = parser.parse_args()
    variants: dict[str, list[dict[str, object]]] = {}
    for variant in ("physical", "normalized"):
        variants[variant] = [_run(ROOT / f"artifacts/temp_compare_{variant}_seed{seed}") for seed in range(3)]
    metric_names = (
        "tasks_completed",
        "net_goal_progress",
        "fraction_steps_progressing_to_goal",
        "actor_goal_projection",
        "oracle_gap",
        "goal_sensitivity_latent_distance",
        "goal_sensitivity_action_distance",
        "alpha",
        "normalized_log_prob",
        "physical_log_prob",
        "mean_Q_oracle_minus_actor",
    )
    aggregate = {
        variant: {
            name: {
                "mean": _mean(runs, name),
                "std": float(np.std([float(run[name]) for run in runs])),
            }
            for name in metric_names
        }
        for variant, runs in variants.items()
    }
    delta = {
        name: aggregate["normalized"][name]["mean"] - aggregate["physical"][name]["mean"]
        for name in metric_names
    }
    normalized = variants["normalized"]
    safety_pass = all(run["safety_pass"] for runs in variants.values() for run in runs)
    positive_signal = sum(float(run["net_goal_progress"]) > 0.0 for run in normalized) >= 2 or any(
        int(run["tasks_completed"]) > 0 for run in normalized
    )
    improvements = {
        "mean_goal_progress_better": delta["net_goal_progress"] > 0.0,
        "actor_goal_projection_better": delta["actor_goal_projection"] > 0.0,
        "oracle_gap_smaller": delta["oracle_gap"] < 0.0,
        "goal_sensitivity_larger": delta["goal_sensitivity_action_distance"] > 0.0,
    }
    count = sum(improvements.values())
    if safety_pass and positive_signal and count == len(improvements):
        classification = "CLEAR_IMPROVEMENT"
    elif safety_pass and positive_signal and count >= 2:
        classification = "WEAK_IMPROVEMENT"
    elif count <= 1:
        classification = "REGRESSION" if delta["net_goal_progress"] < 0.0 else "NO_IMPROVEMENT"
    else:
        classification = "NO_IMPROVEMENT"
    result = {
        "variant_definitions": {
            "physical": "event-only backup reward; alpha adapts using physical action log probability",
            "normalized": "event-only backup reward; alpha adapts using normalized eta log probability",
        },
        "common_config": {
            "scenario": "random_persistent_open",
            "seeds": [0, 1, 2],
            "steps": 2000,
            "warmup_steps": 300,
            "batch_size": 64,
            "target_entropy": -3.0,
            "physical_actor_density": True,
            "physical_bellman_density": True,
        },
        "per_seed": variants,
        "aggregate": aggregate,
        "controlled_delta_normalized_minus_physical": delta,
        "improvement_signals": improvements,
        "positive_task_learning_signal": positive_signal,
        "all_safety_gates_pass": safety_pass,
        "classification": classification,
        "promotion_to_5k": bool(safety_pass and positive_signal and classification in {"CLEAR_IMPROVEMENT", "WEAK_IMPROVEMENT"}),
        "synthetic_only": True,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
