#!/usr/bin/env python3
"""Build the controlled persistent-only versus multi-goal exposure artifact."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "7a0041fc220a4d412c73d6493a8fd8787d2e96bd"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _window(records, low: int, high: int):
    selected = [row for row in records if low <= int(row["step"]) <= high]
    return {
        "steps": len(selected),
        "goal_progress": float(sum(float(row.get("goal_progress", 0.0)) for row in selected)),
        "fraction_progressing": float(np.mean([float(row.get("goal_progress", 0.0)) > 0.0 for row in selected])) if selected else 0.0,
        "tasks_completed": int(sum(bool(row.get("task_completed_now", False)) for row in selected)),
    }


def _training_run(directory: Path):
    summary = _json(directory / "summary.json")
    trajectory = _jsonl(directory / "trajectory_events.jsonl")
    curve = _jsonl(directory / "learning_curve.jsonl")
    metrics = summary["aggregate_metrics"]
    task_distances = [
        float(row["distance_to_goal_after"])
        for row in trajectory
        if row.get("persistent_mode_before") == "TASK_RL"
        and np.isfinite(float(row.get("distance_to_goal_after", np.nan)))
    ]
    safety_fields = (
        "collision_count",
        "energy_depletion_count",
        "uncertified_publication_count",
        "invalid_kappa_fallback_count",
        "fail_closed_steps",
        "accepted_into_kappa_only_count",
    )
    return {
        "directory": str(directory.relative_to(ROOT)),
        "seed": summary["seed"],
        "training_protocol": summary["training_protocol"],
        "goal_exposure_reset_steps": summary["goal_exposure_reset_steps"],
        "unique_goals_seen": summary["goal_exposure"]["unique_goals_seen"],
        "goal_assignments": summary["goal_exposure"]["goal_assignments"],
        "collector_resets": summary["goal_exposure"]["collector_resets"],
        "natural_task_completions": summary["goal_exposure"]["natural_task_completions"],
        "median_steps_per_goal": summary["goal_exposure"]["median_steps_per_goal"],
        "goal_direction_coverage_degrees": summary["goal_exposure"]["goal_direction_angular_coverage_degrees"],
        "mean_logged_batch_unique_goal_count": float(np.mean([row["batch_unique_goal_count"] for row in curve])),
        "tasks_completed": metrics["tasks_completed"],
        "tasks_per_1000_steps": metrics["tasks_per_1000_steps"],
        "net_goal_progress": metrics["total_goal_progress"],
        "fraction_steps_progressing_to_goal": metrics["fraction_steps_progressing_to_goal"],
        "early_0_1000": _window(trajectory, 1, 1000),
        "middle_2000_3000": _window(trajectory, 2001, 3000),
        "late_4000_5000": _window(trajectory, 4001, 5000),
        "minimum_task_rl_distance": None if not task_distances else min(task_distances),
        "rl_generator_fraction": metrics["rl_generator_fraction"],
        "kappa_backup_fraction": metrics["kappa_backup_fraction"],
        "charging_fraction": metrics["charging_fraction"],
        "safety": {field: metrics[field] for field in safety_fields},
    }


def _method_evaluation(results, prefix: str):
    selected = [value for key, value in results.items() if key.startswith(prefix)]
    seed_records = [record for value in selected for record in value["seed_records"]]
    total_steps = sum(value["aggregate_metrics"]["total_steps"] for value in selected)
    total_tasks = sum(value["aggregate_metrics"]["tasks_completed"] for value in selected)
    total_progress = sum(value["aggregate_metrics"]["total_goal_progress"] for value in selected)
    minimum_distances = [record["minimum_task_rl_distance"] for record in seed_records if record["minimum_task_rl_distance"] is not None]
    first_times = [record["time_to_first_completion"] for record in seed_records if record["time_to_first_completion"] is not None]
    safety_fields = (
        "collision_count",
        "energy_depletion_count",
        "uncertified_publication_count",
        "invalid_kappa_fallback_count",
        "fail_closed_steps",
        "accepted_into_kappa_only_count",
    )
    return {
        "checkpoints": len(selected),
        "heldout_runs": len(seed_records),
        "steps": total_steps,
        "tasks_completed": total_tasks,
        "tasks_per_1000_steps": 1000.0 * total_tasks / max(1, total_steps),
        "completion_run_fraction": float(np.mean([record["tasks_completed"] > 0 for record in seed_records])),
        "net_goal_progress": total_progress,
        "mean_progress_per_checkpoint": total_progress / max(1, len(selected)),
        "median_minimum_task_rl_distance": None if not minimum_distances else float(np.median(minimum_distances)),
        "mean_time_to_first_completion": None if not first_times else float(np.mean(first_times)),
        "safety": {
            field: sum(value["aggregate_metrics"][field] for value in selected)
            for field in safety_fields
        },
        "per_checkpoint": selected,
    }


def _goal_conditioning(path: Path):
    audit = _json(path)
    return {
        "physical_states": audit["metadata"]["physical_states"],
        **audit["actor_vs_critic_vs_oracle"],
        "x_reversal_fraction": audit["opposite_goals"]["x_reversal_fraction"],
        "y_reversal_fraction": audit["opposite_goals"]["y_reversal_fraction"],
        "critic_oracle_action_cosine": audit["critic_vs_environment_oracle"]["action_cosine"],
        "classification": audit["PRIMARY_CLASSIFICATION"],
        "gate": audit["COUNTERFACTUAL_GOAL_CRITIC_GATE"],
    }


def main() -> None:
    training = {
        "persistent_only": [
            _training_run(ROOT / f"artifacts/multigoal_5k/persistent_seed{seed}")
            for seed in range(3)
        ],
        "episodic_multi_goal_exposure": [
            _training_run(ROOT / f"artifacts/multigoal_5k/exposure_seed{seed}")
            for seed in range(3)
        ],
    }
    heldout = _json(ROOT / "artifacts/random_persistent/multigoal_heldout_evaluation.json")
    evaluation = {
        "protocol": heldout["evaluation_protocol"],
        "deterministic_actor": heldout["deterministic_actor"],
        "persistent_only": _method_evaluation(heldout["results"], "persistent_"),
        "episodic_multi_goal_exposure": _method_evaluation(heldout["results"], "exposure_"),
    }
    conditioning = {
        "persistent_only": _goal_conditioning(ROOT / "artifacts/random_persistent/counterfactual_goal_critic_persistent_5k.json"),
        "episodic_multi_goal_exposure": _goal_conditioning(ROOT / "artifacts/random_persistent/counterfactual_goal_critic_exposure_5k.json"),
    }
    baseline_progress = np.asarray([row["net_goal_progress"] for row in training["persistent_only"]])
    exposure_progress = np.asarray([row["net_goal_progress"] for row in training["episodic_multi_goal_exposure"]])
    all_safety = [
        value for method in training.values() for row in method for value in row["safety"].values()
    ] + [
        value for method in (evaluation["persistent_only"], evaluation["episodic_multi_goal_exposure"])
        for value in method["safety"].values()
    ]
    payload = {
        "baseline_commit": BASELINE_COMMIT,
        "implementation_commit": None,
        "implementation_revision": "uncommitted working tree based on baseline commit",
        "training_configs": {
            "common": {
                "scenario": "random_persistent_open",
                "seeds": [0, 1, 2],
                "steps": 5000,
                "warmup_steps": 300,
                "batch_size": 64,
                "temperature_coordinate": "physical",
            },
            "persistent_only": {"goal_exposure_reset_steps": None},
            "episodic_multi_goal_exposure": {"goal_exposure_reset_steps": 250},
        },
        "implementation": {
            "reset_boundary_representation": "independent collector_boundary replay field",
            "bootstrap_across_reset_boundary": False,
            "agent_preserved": True,
            "replay_preserved": True,
            "persistent_evaluation_semantics_preserved": True,
        },
        "training": training,
        "heldout_persistent_evaluation": evaluation,
        "goal_conditioning": conditioning,
        "effect_summaries": {
            "mean_training_progress_persistent": float(np.mean(baseline_progress)),
            "mean_training_progress_exposure": float(np.mean(exposure_progress)),
            "mean_training_progress_delta": float(np.mean(exposure_progress) - np.mean(baseline_progress)),
            "exposure_better_training_progress_seeds": int(np.sum(exposure_progress > baseline_progress)),
            "heldout_mean_progress_delta_per_checkpoint": (
                evaluation["episodic_multi_goal_exposure"]["mean_progress_per_checkpoint"]
                - evaluation["persistent_only"]["mean_progress_per_checkpoint"]
            ),
            "S_actor_delta": conditioning["episodic_multi_goal_exposure"]["S_actor"] - conditioning["persistent_only"]["S_actor"],
            "S_Q_delta": conditioning["episodic_multi_goal_exposure"]["S_Q"] - conditioning["persistent_only"]["S_Q"],
        },
        "facts": [
            "Each exposure run saw 20 unique goals and 19 collector resets; each persistent-only run saw one goal.",
            "Exposure training progress was positive in all three seeds; persistent-only progress was negative in all three seeds.",
            "Neither condition completed a task during training or held-out persistent evaluation.",
            "Held-out progress remained negative for every checkpoint, although exposure had a small aggregate advantage.",
            "Counterfactual actor and critic preference sensitivity did not improve under exposure; opposite-goal reversal remained zero.",
            "All recorded training and held-out safety counts were zero.",
        ],
        "inferences": [
            "Goal exposure materially changes the training data distribution and improves on-policy training progress, but 5k does not establish persistent task competence.",
            "The lack of improved counterfactual sensitivity means the positive training-progress effect should not be interpreted as closed goal-conditioned learning.",
        ],
        "alternative_explanations": {
            "training_length": "Equal-step 5k persistent-only controls this comparison and remained weak.",
            "goal_exposure": "Consistent positive training-progress delta, but no completions or goal-sensitivity restoration.",
            "critic_learning": "Both 5k audits remain CRITIC_GOAL_INSENSITIVE.",
            "temporal_credit": "Not changed or tested in this controlled experiment.",
            "entropy_scale": "Not changed or tested in this controlled experiment.",
        },
        "safety_regression": bool(any(all_safety)),
        "primary_classification": "GOAL_EXPOSURE_NOT_SUFFICIENT",
        "promotion_to_multi_method_10k_pilot": False,
        "promotion_rationale": "Exposure improved collection-time progress in 3/3 seeds, but held-out persistent progress remained negative, completions and reversal stayed zero, goal sensitivities decreased, and late windows did not improve over early windows. Task-learning evidence is therefore insufficient for a 10k multi-method pilot.",
        "promotion_to_50k": False,
        "exact_next_action": "In the next phase, keep the 250-step exposure protocol fixed and test exactly one temporal-credit intervention: a single predeclared n-step Generator-SAC target against the one-step exposure baseline in a controlled 3-seed short pilot.",
    }
    output = ROOT / "artifacts/random_persistent/multigoal_training_comparison.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["effect_summaries"], indent=2))


if __name__ == "__main__":
    main()
