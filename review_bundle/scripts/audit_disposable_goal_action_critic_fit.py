#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cert_runtime.bellman_goal_action_diagnostics import finite_preferred_actions, mean_pairwise_action_distance
from cert_runtime.counterfactual_goal_diagnostics import residual_alignment
from cert_runtime.crossed_horizon_diagnostics import fit_disposable_critic


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _dataset(cases, horizon, semantics, coverage):
    target_name = {
        "physical": "physical_target",
        "no_entropy": "no_entropy_target",
        "normalized_entropy": "normalized_entropy_target",
    }[semantics]
    observations = []
    actions = []
    targets = []
    for case in cases:
        observation_matrix = np.asarray(case["observations"], dtype=np.float32)
        action_matrix = np.asarray(case["actions"], dtype=np.float32)
        target_matrix = np.asarray(case["horizons"][str(horizon)][target_name], dtype=np.float32)
        target_matrix = (target_matrix - target_matrix.mean(axis=1, keepdims=True)) / np.maximum(
            target_matrix.std(axis=1, keepdims=True), 1e-6
        )
        goal_indices = range(len(observation_matrix)) if coverage == "counterfactual" else (0,)
        for goal_index in goal_indices:
            observations.extend(np.repeat(observation_matrix[goal_index:goal_index + 1], len(action_matrix), axis=0))
            actions.extend(action_matrix)
            targets.extend(target_matrix[goal_index])
    return np.asarray(observations), np.asarray(actions), np.asarray(targets)


def _evaluate(network, cases, horizon, semantics):
    target_name = {
        "physical": "physical_target",
        "no_entropy": "no_entropy_target",
        "normalized_entropy": "normalized_entropy_target",
    }[semantics]
    sensitivities = []
    target_sensitivities = []
    alignments = []
    reversals_x = []
    reversals_y = []
    mse = []
    ranking = []
    for case in cases:
        observations = np.asarray(case["observations"], dtype=np.float32)
        actions = np.asarray(case["actions"], dtype=np.float32)
        target = np.asarray(case["horizons"][str(horizon)][target_name], dtype=np.float64)
        normalized_target = (target - target.mean(axis=1, keepdims=True)) / np.maximum(
            target.std(axis=1, keepdims=True), 1e-6
        )
        prediction = np.stack([
            network(
                torch.as_tensor(np.repeat(observations[index:index + 1], len(actions), axis=0)),
                torch.as_tensor(actions),
            ).detach().numpy()
            for index in range(len(observations))
        ])
        _, preferred = finite_preferred_actions(prediction, actions)
        _, target_preferred = finite_preferred_actions(normalized_target, actions)
        oracle = np.asarray(case["environment_oracle_actions"], dtype=np.float64)
        center = np.asarray(case["c"], dtype=np.float64)
        labels = [item["label"] for item in case["goals"]]
        sensitivities.append(mean_pairwise_action_distance(preferred))
        target_sensitivities.append(mean_pairwise_action_distance(target_preferred))
        alignments.extend(residual_alignment(preferred[index], oracle[index], center) for index in range(len(preferred)))
        mse.append(float(np.mean((prediction - normalized_target) ** 2)))
        ranking.append(float(np.mean(np.argmax(prediction, axis=1) == np.argmax(normalized_target, axis=1))))
        for axis, output in ((0, reversals_x), (1, reversals_y)):
            positive_label, negative_label = (("+x", "-x") if axis == 0 else ("+y", "-y"))
            positive, negative = labels.index(positive_label), labels.index(negative_label)
            output.append(bool(np.sign(preferred[positive, axis]) != np.sign(preferred[negative, axis])))
    return {
        "heldout_S_Q_probe": float(np.mean(sensitivities)),
        "heldout_S_target": float(np.mean(target_sensitivities)),
        "x_reversal_fraction": float(np.mean(reversals_x)),
        "y_reversal_fraction": float(np.mean(reversals_y)),
        "oracle_alignment_mean": float(np.mean(alignments)),
        "target_mse": float(np.mean(mse)),
        "preferred_action_ranking_accuracy": float(np.mean(ranking)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/random_persistent/crossed_horizon_goal_coverage_representative_cases.json")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--target-semantics", choices=("physical", "no_entropy", "normalized_entropy"), default="physical")
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--output", default="artifacts/random_persistent/disposable_critic_identifiability_probe.json")
    args = parser.parse_args()
    cases = json.loads((ROOT / args.input).read_text())["representative_cases"]
    results = {}
    for seed, coverage in enumerate(("actual", "counterfactual")):
        observations, actions, targets = _dataset(cases, args.horizon, args.target_semantics, coverage)
        network, fit = fit_disposable_critic(
            observations, actions, targets, hidden_dim=128, steps=args.steps, seed=900 + seed
        )
        results[coverage] = {
            "dataset_rows": len(observations),
            "order_preserving_target_transform": "per-state-goal action advantage standardization",
            "fit": fit,
            "heldout_counterfactual_grid": _evaluate(network, cases, args.horizon, args.target_semantics),
        }
    counterfactual = results["counterfactual"]["heldout_counterfactual_grid"]
    target_sensitivity = counterfactual["heldout_S_target"]
    recovered = counterfactual["heldout_S_Q_probe"] / max(target_sensitivity, 1e-12)
    representation_failure = bool(recovered < 0.25 and counterfactual["preferred_action_ranking_accuracy"] < 0.5)
    output = {
        "metadata": {
            "architecture": "current QNetwork concat(observation, physical_action), hidden_dim=128",
            "horizon": args.horizon,
            "target_semantics": args.target_semantics,
            "DIAGNOSTIC_TEMPORARY_MODEL_ONLY": True,
            "ENVIRONMENT_TRAINING_STEPS": 0,
            "PRODUCTION_ACTOR_UPDATED": False,
            "PRODUCTION_CRITIC_UPDATED": False,
            "TARGET_CRITIC_UPDATED": False,
            "REPLAY_MODIFIED": False,
        },
        "results": results,
        "counterfactual_preference_recovery_ratio": recovered,
        "representation_failure_supported": representation_failure,
    }
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, default=_json_default) + "\n")
    crossed_path = ROOT / "artifacts/random_persistent/crossed_horizon_goal_coverage_audit.json"
    if crossed_path.exists():
        crossed = json.loads(crossed_path.read_text())
        crossed["disposable_critic_probe"] = {
            "artifact": str(output_path.relative_to(ROOT)),
            "counterfactual_preference_recovery_ratio": recovered,
            "counterfactual_target_mse": counterfactual["target_mse"],
            "counterfactual_preferred_action_ranking_accuracy": counterfactual["preferred_action_ranking_accuracy"],
            "representation_failure_supported": representation_failure,
        }
        if representation_failure:
            crossed["PRIMARY_CLASSIFICATION"] = "CRITIC_REPRESENTATION_FAILURE"
            crossed["NEXT_SINGLE_CANDIDATE"] = "GOAL_ACTION_INTERACTION_CRITIC"
        crossed_path.write_text(json.dumps(crossed, indent=2, default=_json_default) + "\n")
    print(f"CRITIC_REPRESENTATION_FAILURE_SUPPORTED = {'YES' if representation_failure else 'NO'}")


if __name__ == "__main__":
    main()
