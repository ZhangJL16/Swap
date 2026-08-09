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

from cert_runtime.replay_goal_action_diagnostics import (
    action_covariance_metrics,
    counterfactual_augmented_features,
    effective_rank,
    goal_action_interaction_features,
    goal_direction_diversity,
    knn_physical_neighborhoods,
    observed_action_support_coverage,
    physical_feature,
    unit_directions,
)
from envs.certified_uav import make_random_persistent_uav_env


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _stats(values) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {name: float("nan") for name in ("mean", "median", "p10", "p90", "min", "max")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _load_rows(paths):
    rows = []
    for seed, path in enumerate(paths):
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()):
            if (
                row.get("execution_authority") == "RL_GENERATOR"
                and row.get("accepted") is True
                and row.get("goal") is not None
                and row.get("actor_eta") is not None
                and row.get("rl_authority_set_member") is not False
            ):
                selected = dict(row)
                selected["checkpoint_seed"] = seed
                selected["position"] = row.get("position_before", row["position"])
                selected["velocity"] = row.get("velocity_before", row["velocity"])
                selected["energy"] = row.get("energy_before", row["energy"])
                rows.append(selected)
    return rows


def _correlation_norm(goals, actions):
    joined = np.concatenate((goals, actions), axis=1)
    if len(joined) < 3:
        return 0.0
    correlation = np.corrcoef(joined, rowvar=False)[:3, 3:]
    return float(np.linalg.norm(np.nan_to_num(correlation), ord="fro"))


def _n_step_signal(paths, gamma):
    result = {}
    for horizon in (1, 3, 5, 10):
        cumulative = []
        alignments = []
        for path in paths:
            raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            for start in range(0, max(0, len(raw) - horizon + 1)):
                segment = raw[start:start + horizon]
                first = segment[0]
                if not all(
                    row.get("episode_id") == first.get("episode_id")
                    and row.get("task_id") == first.get("task_id")
                    and row.get("task_completed_now") is False
                    and row.get("charging") is False
                    for row in segment
                ):
                    continue
                rewards = [float(row.get("reward_components", {}).get("goal_progress_reward", 0.0)) for row in segment]
                cumulative.append(float(sum((gamma ** offset) * reward for offset, reward in enumerate(rewards))))
                position = np.asarray(first.get("position_before", first["position"]), dtype=np.float64)
                goal = np.asarray(first["goal"], dtype=np.float64)
                direction = goal - position
                direction /= max(float(np.linalg.norm(direction)), 1e-12)
                action = np.asarray(first["executed_action"], dtype=np.float64)
                alignments.append(float(action @ direction))
        if cumulative:
            values = np.asarray(cumulative)
            alignment_values = np.asarray(alignments)
            lower = values[alignment_values <= np.percentile(alignment_values, 25)]
            upper = values[alignment_values >= np.percentile(alignment_values, 75)]
            contrast = float(np.mean(upper) - np.mean(lower)) if len(lower) and len(upper) else 0.0
        else:
            contrast = 0.0
        result[str(horizon)] = {
            "segments": len(cumulative),
            "cumulative_goal_progress_signal": _stats(cumulative),
            "high_vs_low_initial_goal_alignment_contrast": contrast,
        }
    return result


def _classification(bellman, replay_gate, interaction_gain):
    target = bellman["target_preference"]
    interaction = bellman["bellman_target"]["interaction_preservation_ratio"]["mean"]
    target_over_oracle = float(target["S_Y"]) / max(float(target["S_oracle"]), 1e-12)
    critic_interaction = bellman["additive_decomposition"]["learned_critic"]["interaction_variance"]["mean"]
    target_interaction = bellman["additive_decomposition"]["bellman_target"]["interaction_variance"]["mean"]
    target_reversal = float(target.get("target_reversal_mean", 0.0))
    bellman_weak = (
        bellman["BELLMAN_GOAL_ACTION_COUPLING_GATE"] == "FAIL"
        and target_over_oracle < 0.05
        and target_reversal < 0.1
    )
    replay_unidentifiable = replay_gate == "FAIL" and interaction_gain > 0.5
    if bellman_weak and replay_unidentifiable:
        return "MIXED", "one-step targets suppress goal-conditioned preference while local replay also lacks goal-direction coverage"
    if bellman_weak:
        return "BELLMAN_TARGET_GOAL_WEAK", "one-step targets contain interaction variance but suppress the immediate goal-conditioned preferred action"
    if bellman["BELLMAN_GOAL_ACTION_COUPLING_GATE"] in {"PASS", "MARGINAL"} and replay_gate == "FAIL" and interaction_gain > 0.5:
        return "REPLAY_GOAL_ACTION_UNIDENTIFIABLE", "counterfactual pairing materially increases interaction rank over local rollout replay"
    if (
        bellman["BELLMAN_GOAL_ACTION_COUPLING_GATE"] == "PASS"
        and replay_gate == "PASS"
        and critic_interaction < 0.1 * max(target_interaction, 1e-12)
    ):
        return "CRITIC_REPRESENTATION_FAILURE", "targets and replay identify interaction but learned critic remains nearly additive"
    return "MIXED", "Bellman coupling, replay coverage, and critic fitting provide overlapping or incomplete evidence"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", nargs="+", default=[f"artifacts/temp_compare_physical_seed{seed}/trajectory_events.jsonl" for seed in range(3)])
    parser.add_argument("--scenario", default="random_persistent_open")
    parser.add_argument("--bellman-audit", default="artifacts/random_persistent/bellman_goal_action_coupling_audit.json")
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--anchors-per-cell", type=int, default=8)
    parser.add_argument("--output", default="artifacts/random_persistent/replay_goal_action_identifiability_audit.json")
    args = parser.parse_args()
    paths = [ROOT / path for path in args.trajectory]
    rows = _load_rows(paths)
    environment = make_random_persistent_uav_env(f"{args.scenario}.json", seed=0)
    features = np.stack([
        physical_feature(row, environment.plant.config.world_size, environment.plant.config.v_max, environment.task_env.battery_capacity)
        for row in rows
    ])
    neighborhoods = knn_physical_neighborhoods(
        rows,
        features,
        neighbors=args.neighbors,
        anchors_per_cell=args.anchors_per_cell,
    )
    canonical_goals = np.asarray((
        (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
        (1.0, 1.0, 0.0), (-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (-1.0, 1.0, 0.0),
    ))
    records = []
    for neighborhood_id, indices in enumerate(neighborhoods):
        selected = [rows[int(index)] for index in indices]
        positions = np.stack([np.asarray(row["position"], dtype=np.float64) for row in selected])
        goals = np.stack([np.asarray(row["goal"], dtype=np.float64) for row in selected])
        goal_directions = unit_directions(goals - positions)
        etas = np.stack([np.asarray(row["actor_eta"], dtype=np.float64) for row in selected])
        actions = np.stack([np.asarray(row["executed_action"], dtype=np.float64) for row in selected])
        diversity = goal_direction_diversity(goal_directions)
        covariance = action_covariance_metrics(actions)
        actual_features = goal_action_interaction_features(goal_directions, etas)
        actual_rank = effective_rank(actual_features)
        augmented_rank = effective_rank(counterfactual_augmented_features(canonical_goals, etas))
        support_coverage = observed_action_support_coverage(etas)
        record = {
            "neighborhood_id": neighborhood_id,
            "cell_id": selected[0].get("kappa_cell_id"),
            "mode": selected[0].get("persistent_mode"),
            "transitions": len(selected),
            "distinct_goals": len({tuple(np.round(np.asarray(row["goal"], dtype=np.float64), 4)) for row in selected}),
            **diversity,
            "action_covariance_rank": covariance["rank"],
            "action_covariance_eigenvalues": covariance["eigenvalues"],
            "interaction_rank": actual_rank["rank"],
            "interaction_effective_rank": actual_rank["effective_rank"],
            "interaction_smallest_singular": actual_rank["smallest_nonzero_singular_value"],
            "interaction_condition_number": actual_rank["condition_number"],
            "counterfactual_interaction_rank": augmented_rank["rank"],
            "counterfactual_interaction_effective_rank": augmented_rank["effective_rank"],
            "counterfactual_information_gain": float(augmented_rank["effective_rank"] - actual_rank["effective_rank"]),
            "observed_support_action_coverage": support_coverage["mean_std_over_half_width"],
            "goal_action_correlation_norm": _correlation_norm(goal_directions, etas),
            "reward_variance": float(np.var([float(row.get("reward", 0.0)) for row in selected])),
            "goal_progress_variance": float(np.var([float(row.get("goal_progress", 0.0)) for row in selected])),
        }
        records.append(record)
    values = lambda name: [float(record[name]) for record in records]
    near_opposite_fraction = float(np.mean([record["near_opposite"] for record in records])) if records else 0.0
    full_rank_fraction = float(np.mean(np.asarray(values("interaction_rank")) == 9)) if records else 0.0
    median_effective_rank = float(np.median(values("interaction_effective_rank"))) if records else 0.0
    median_coverage = float(np.median(values("observed_support_action_coverage"))) if records else 0.0
    median_distinct = float(np.median(values("distinct_directions"))) if records else 0.0
    strong = near_opposite_fraction >= 0.5 and median_effective_rank >= 4.5 and median_coverage >= 0.1 and median_distinct >= 4
    partial = (
        median_distinct >= 2
        and median_effective_rank >= 2.0
        and median_coverage >= 0.05
    ) or near_opposite_fraction >= 0.2
    replay_gate = "PASS" if strong else ("MARGINAL" if partial else "FAIL")
    bellman = json.loads((ROOT / args.bellman_audit).read_text(encoding="utf-8"))
    information_gain = float(np.mean(values("counterfactual_information_gain"))) if records else 0.0
    classification, rationale = _classification(bellman, replay_gate, information_gain)
    output = {
        "metadata": {
            "trajectories": [str(path.relative_to(ROOT)) for path in paths],
            "eligible_transitions": len(rows),
            "local_neighborhoods_analyzed": len(records),
            "neighborhood_method": "same recovery cell and mode, deterministic k-nearest physical-state neighborhoods",
            "neighbors": args.neighbors,
            "training_steps": 0,
            "actor_updated": False,
            "critic_updated": False,
            "replay_modified": False,
            "counterfactual_augmentation_diagnostic_only": True,
            "safety_support_modified": False,
            "reward_modified": False,
        },
        "local_goal_diversity": {
            "transitions_per_neighborhood": _stats(values("transitions")),
            "distinct_goal_directions": _stats(values("distinct_directions")),
            "goal_angular_spread_degrees": _stats(values("angular_spread_degrees")),
            "goal_direction_entropy": _stats(values("direction_entropy")),
            "near_opposite_goal_neighborhood_fraction": near_opposite_fraction,
        },
        "local_action_diversity": {
            "action_covariance_rank": _stats(values("action_covariance_rank")),
            "observed_over_available_support_coverage": _stats(values("observed_support_action_coverage")),
            "goal_action_correlation_norm": _stats(values("goal_action_correlation_norm")),
        },
        "interaction_identifiability": {
            "full_interaction_rank_fraction": full_rank_fraction,
            "interaction_rank": _stats(values("interaction_rank")),
            "interaction_effective_rank": _stats(values("interaction_effective_rank")),
            "smallest_nonzero_singular_value": _stats(values("interaction_smallest_singular")),
            "condition_number": _stats(value for value in values("interaction_condition_number") if np.isfinite(value)),
            "reward_variance": _stats(values("reward_variance")),
            "goal_progress_variance": _stats(values("goal_progress_variance")),
        },
        "counterfactual_relabel_diagnostic": {
            "actual_interaction_rank": _stats(values("interaction_rank")),
            "actual_effective_rank": _stats(values("interaction_effective_rank")),
            "counterfactual_augmented_interaction_rank": _stats(values("counterfactual_interaction_rank")),
            "counterfactual_augmented_effective_rank": _stats(values("counterfactual_interaction_effective_rank")),
            "effective_rank_information_gain": _stats(values("counterfactual_information_gain")),
            "relabel_inserted_into_replay": False,
            "physical_successor_invented": False,
        },
        "n_step_offline_signal": _n_step_signal(paths, 0.99),
        "REPLAY_GOAL_ACTION_IDENTIFIABILITY_GATE": replay_gate,
        "gate_rationale": (
            "local replay provides broad independent goal-action interaction coverage"
            if strong else
            "local replay provides partial but incomplete goal-action interaction coverage"
            if partial else
            "local replay neighborhoods do not identify broad goal-action interactions"
        ),
        "PRIMARY_CLASSIFICATION": classification,
        "root_cause_interpretation": rationale,
        "NEXT_SINGLE_CANDIDATE": {
            "BELLMAN_TARGET_GOAL_WEAK": "one n-step Generator-SAC target candidate",
            "REPLAY_GOAL_ACTION_UNIDENTIFIABLE": "strict noncompletion counterfactual goal relabeling for critic training",
            "CRITIC_REPRESENTATION_FAILURE": "explicit goal-action interaction critic",
            "MIXED": "refine the strongest Bellman/replay/representation contribution before intervention",
        }[classification],
        "neighborhood_records": records,
        "synthetic_only": True,
    }
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"REPLAY_GOAL_ACTION_IDENTIFIABILITY_GATE = {replay_gate}")
    print(f"PRIMARY_CLASSIFICATION = {classification}")
    print(json.dumps({key: output[key] for key in ("metadata", "local_goal_diversity", "local_action_diversity", "interaction_identifiability", "counterfactual_relabel_diagnostic", "n_step_offline_signal", "gate_rationale", "root_cause_interpretation", "NEXT_SINGLE_CANDIDATE")}, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
