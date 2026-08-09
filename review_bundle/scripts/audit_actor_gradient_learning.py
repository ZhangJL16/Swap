#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cert_runtime.actor import FeedForwardAffineTanhActor
from cert_runtime.actor_gradient_diagnostics import (
    EPSILON,
    action_to_latent_gradient,
    actor_goal_jacobians,
    actor_gradient_decomposition,
    critic_action_column_statistics,
    critic_action_gradient,
    critic_goal_jacobian,
    directional_finite_difference,
    first_layer_column_statistics,
    interpolation_landscape,
    q_through_actor_goal_gradient,
    scalar_statistics,
)
from cert_runtime.generator_sac import GeneratorSACConfig, PersistentGeneratorSAC
from cert_runtime.optimization_diagnostics import entropy_decomposition, goal_projection_metrics
from cert_runtime.task_authority import BestInGeneratorGoalOracle, action_from_eta
from envs.certified_uav import make_random_persistent_uav_env
from envs.certified_uav.persistent_task import PersistentMissionMode
from envs.certified_uav.state import UAVPhysicalState


def _load_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _restore(environment, row: dict[str, object]) -> tuple[np.ndarray, dict[str, object]]:
    environment.plant.state = UAVPhysicalState(
        np.asarray(row["position"], dtype=np.float64),
        np.asarray(row["velocity"], dtype=np.float64),
        float(row["energy"]),
        0.0,
    )
    environment.plant.failure_reason = None
    environment.plant.last_lidar = environment.plant.lidar_model.measure(
        environment.plant.state, environment.plant.world, environment.plant.np_random
    )
    environment.task_env.manager.current_task.goal_position = np.asarray(row["goal"], dtype=np.float64)
    environment.task_env.mode = PersistentMissionMode.TASK_RL
    environment.task_env.phase = environment.task_env.mode
    environment._context_cache_key = None
    context = environment._refresh_context()
    observation = environment.task_env.build_observation(
        environment.runtime._map_encoding(), environment.runtime._corridor_encoding()
    )
    return observation, context


def _q_min(agent, observations, actions):
    return torch.minimum(agent.critic_1(observations, actions), agent.critic_2(observations, actions))


def _goal_jacobian_for_actor(actor, observations, centers, generators, goal_slice) -> float:
    values = []
    for index in range(min(20, observations.shape[0])):
        selected = observations[index:index + 1].detach().clone().requires_grad_(True)
        mean = actor.distribution(selected).mean
        action = centers[index:index + 1] + torch.bmm(
            generators[index:index + 1], torch.tanh(mean).unsqueeze(-1)
        ).squeeze(-1)
        rows = []
        for axis in range(3):
            rows.append(torch.autograd.grad(action[:, axis].sum(), selected, retain_graph=True)[0][:, goal_slice])
        values.append(float(torch.linalg.matrix_norm(torch.stack(rows, dim=1)).detach()))
    return float(np.mean(values))


def _evaluate_actor(actor, agent, data, goal_slice) -> dict[str, float]:
    observations = data["observations"]
    centers = data["centers"]
    generators = data["generators"]
    with torch.no_grad():
        mean = actor.distribution(observations).mean
        actions = centers + torch.bmm(generators, torch.tanh(mean).unsqueeze(-1)).squeeze(-1)
        q_actor = _q_min(agent, observations, actions)
    projections = []
    gaps = []
    for index, action in enumerate(actions.cpu().numpy()):
        metrics = goal_projection_metrics(
            data["positions"][index],
            data["velocities"][index],
            data["goals"][index],
            action,
            centers[index].cpu().numpy(),
            data["oracle_actions"][index],
            data["dt"],
        )
        projections.append(metrics["actor_goal_projection"])
        gaps.append(metrics["oracle_gap"])
    distribution = actor.distribution(observations)
    deterministic_actions = centers + torch.bmm(
        generators, torch.tanh(distribution.mean).unsqueeze(-1)
    ).squeeze(-1)
    q_loss = -_q_min(agent, observations, deterministic_actions).mean()
    gradients = torch.autograd.grad(q_loss, list(actor.parameters()), allow_unused=True)
    q_gradient_norm = float(torch.sqrt(sum(
        torch.zeros((), dtype=q_loss.dtype) if gradient is None else gradient.square().sum()
        for gradient in gradients
    )))
    return {
        "Q_actor": float(q_actor.mean()),
        "actor_goal_projection": float(np.mean(projections)),
        "oracle_gap": float(np.mean(gaps)),
        "action_goal_jacobian_norm": _goal_jacobian_for_actor(
            actor, observations, centers, generators, goal_slice
        ),
        "Q_gradient_norm": q_gradient_norm,
    }


def _frozen_actor_updates(agent, data, goal_slice, seed: int, updates: int = 500) -> dict[str, object]:
    for critic in (agent.critic_1, agent.critic_2, agent.target_critic_1, agent.target_critic_2):
        critic.eval()
        for parameter in critic.parameters():
            parameter.requires_grad_(False)
    critic_before = [parameter.detach().clone() for critic in (agent.critic_1, agent.critic_2) for parameter in critic.parameters()]
    actors = {"Q_ONLY": deepcopy(agent.actor), "CURRENT_ACTOR_OBJECTIVE": deepcopy(agent.actor)}
    optimizers = {
        name: torch.optim.Adam(actor.parameters(), lr=agent.config.actor_lr)
        for name, actor in actors.items()
    }
    rng = np.random.default_rng(seed + 9000)
    torch_generator = torch.Generator().manual_seed(seed + 12000)
    batch_size = min(agent.config.batch_size, data["observations"].shape[0])
    records = {name: [{"update": 0, **_evaluate_actor(actor, agent, data, goal_slice)}] for name, actor in actors.items()}
    for update in range(1, updates + 1):
        indices = torch.as_tensor(
            rng.choice(data["observations"].shape[0], size=batch_size, replace=False), dtype=torch.long
        )
        noise = torch.randn((batch_size, 3), generator=torch_generator)
        for name, actor in actors.items():
            observations = data["observations"][indices]
            centers = data["centers"][indices]
            generators = data["generators"][indices]
            distribution = actor.distribution(observations)
            u = distribution.mean + distribution.stddev * noise
            actions = centers + torch.bmm(generators, torch.tanh(u).unsqueeze(-1)).squeeze(-1)
            q_value = _q_min(agent, observations, actions)
            if name == "Q_ONLY":
                loss = -q_value.mean()
            else:
                physical_log_probability = entropy_decomposition(
                    distribution, u, generators
                ).physical_log_prob
                loss = (agent.alpha.detach() * physical_log_probability - q_value).mean()
            optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            optimizers[name].step()
        if update % 50 == 0:
            for name, actor in actors.items():
                records[name].append({"update": update, **_evaluate_actor(actor, agent, data, goal_slice)})
    critic_after = [parameter.detach().clone() for critic in (agent.critic_1, agent.critic_2) for parameter in critic.parameters()]
    return {
        "updates": updates,
        "fixed_recorded_state_stream": True,
        "oracle_used_for_training": False,
        "critic_parameters_unchanged": all(torch.equal(before, after) for before, after in zip(critic_before, critic_after)),
        "trajectories": records,
    }


def _checkpoint_audit(checkpoint_path: Path, scenario: str, sample_count: int, seed: int):
    directory = checkpoint_path.parent
    trajectory_path = directory / "trajectory_events.jsonl"
    environment = make_random_persistent_uav_env(f"{scenario}.json", seed=seed)
    observation, _ = environment.reset(seed=seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = GeneratorSACConfig(**checkpoint["config"])
    untrained = PersistentGeneratorSAC(observation.size, config, seed=seed)
    untrained_actor = deepcopy(untrained.actor)
    agent = PersistentGeneratorSAC(observation.size, config, seed=seed)
    agent.load_state_dict(checkpoint)
    agent.actor.eval()
    agent.critic_1.eval()
    agent.critic_2.eval()
    rows = [
        row for row in _load_rows(trajectory_path)
        if row.get("execution_authority") == "RL_GENERATOR" and row.get("goal") is not None
    ]
    indices = np.linspace(0, len(rows) - 1, min(sample_count, len(rows)), dtype=int)
    selected_rows = [rows[index] for index in indices]
    records = []
    oracle = BestInGeneratorGoalOracle()
    for row in selected_rows:
        selected_observation, context = _restore(environment, row)
        if not context.get("generator_executable"):
            continue
        center = np.asarray(context["c"], dtype=np.float64)
        generator = np.asarray(context["G"], dtype=np.float64)
        state = environment.plant.state.copy()
        goal = np.asarray(row["goal"], dtype=np.float64)
        observation_tensor = torch.as_tensor(selected_observation, dtype=torch.float32).unsqueeze(0)
        center_tensor = torch.as_tensor(center, dtype=torch.float32).unsqueeze(0)
        generator_tensor = torch.as_tensor(generator, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean_u = agent.actor.distribution(observation_tensor).mean
            actor_action_tensor = center_tensor + torch.bmm(
                generator_tensor, torch.tanh(mean_u).unsqueeze(-1)
            ).squeeze(-1)
        actor_action = actor_action_tensor[0].numpy()
        oracle_eta = oracle.select_eta(state, goal, center, generator, environment.plant.config.dt)
        oracle_action = action_from_eta(center, generator, oracle_eta)
        oracle_tensor = torch.as_tensor(oracle_action, dtype=torch.float32).unsqueeze(0)
        center_action_tensor = center_tensor
        q_actor, grad_action = critic_action_gradient(agent, observation_tensor, actor_action_tensor)
        with torch.no_grad():
            q_oracle = _q_min(agent, observation_tensor, oracle_tensor)
            q_center = _q_min(agent, observation_tensor, center_action_tensor)
        direction = oracle_tensor - actor_action_tensor
        direction = direction / (torch.linalg.vector_norm(direction, dim=-1, keepdim=True) + EPSILON)
        directional = (grad_action * direction).sum(-1)
        finite_difference = directional_finite_difference(
            agent, observation_tensor, actor_action_tensor, direction
        )
        transmission = action_to_latent_gradient(grad_action, generator_tensor, mean_u)
        mean_jacobian, action_jacobian = actor_goal_jacobians(
            agent,
            observation_tensor,
            center_tensor,
            generator_tensor,
            environment.task_env.observation_layout["goal_delta"],
        )
        critic_jacobian = critic_goal_jacobian(
            agent,
            observation_tensor,
            actor_action_tensor,
            environment.task_env.observation_layout["goal_delta"],
        )
        q_actor_goal_gradient = q_through_actor_goal_gradient(
            agent,
            observation_tensor,
            center_tensor,
            generator_tensor,
            environment.task_env.observation_layout["goal_delta"],
        )
        landscape = interpolation_landscape(
            agent, observation_tensor, actor_action_tensor, oracle_tensor
        )
        records.append({
            "step": int(row.get("step", 0)),
            "observation": selected_observation,
            "center": center,
            "generator": generator,
            "position": state.position.copy(),
            "velocity": state.velocity.copy(),
            "goal": goal,
            "oracle_action": oracle_action,
            "actor_action": actor_action,
            "mean_u": mean_u[0].numpy(),
            "Q_actor": float(q_actor),
            "Q_oracle": float(q_oracle),
            "Q_center": float(q_center),
            "grad_action": grad_action[0].numpy(),
            "directional": float(directional),
            "finite_difference": float(finite_difference),
            "transmission": transmission,
            "mean_goal_jacobian_norm": float(torch.linalg.matrix_norm(mean_jacobian)),
            "action_goal_jacobian_norm": float(torch.linalg.matrix_norm(action_jacobian)),
            "critic_goal_jacobian_norm": float(torch.linalg.vector_norm(critic_jacobian)),
            "Q_actor_goal_gradient_norm": float(torch.linalg.vector_norm(q_actor_goal_gradient)),
            "landscape": landscape,
        })
    observations = torch.as_tensor(np.stack([record["observation"] for record in records]), dtype=torch.float32)
    centers = torch.as_tensor(np.stack([record["center"] for record in records]), dtype=torch.float32)
    generators = torch.as_tensor(np.stack([record["generator"] for record in records]), dtype=torch.float32)
    positions = np.stack([record["position"] for record in records])
    velocities = np.stack([record["velocity"] for record in records])
    goals = np.stack([record["goal"] for record in records])
    oracle_actions = np.stack([record["oracle_action"] for record in records])
    noise = torch.randn((observations.shape[0], 3), generator=torch.Generator().manual_seed(seed + 3000))
    gradient_decomposition = actor_gradient_decomposition(agent, observations, centers, generators, noise)
    goal_slice = environment.task_env.observation_layout["goal_delta"]
    trained_goal_jacobian = float(np.mean([record["action_goal_jacobian_norm"] for record in records]))
    untrained_goal_jacobian = _goal_jacobian_for_actor(
        untrained_actor, observations, centers, generators, goal_slice
    )
    transmission_rows = [record["transmission"] for record in records]
    landscape_counts = {
        name: sum(record["landscape"]["classification"] == name for record in records)
        for name in (
            "MONOTONIC_TOWARD_ORACLE",
            "INITIAL_POSITIVE_SLOPE",
            "FLAT_NEAR_ACTOR",
            "LOCAL_WRONG_DIRECTION",
            "NONMONOTONIC",
        )
    }
    action_values = np.stack([record["actor_action"] for record in records])
    observation_std = np.std(observations.numpy(), axis=0)
    representative_observation_std = float(np.median(observation_std[observation_std > 1e-8]))
    grad_action_norms = [float(np.linalg.norm(record["grad_action"])) for record in records]
    action_max = np.asarray(environment.plant.config.a_max, dtype=np.float64)
    hypothetical_norms = [
        float(np.linalg.norm(action_max * record["grad_action"])) for record in records
    ]
    actor_groups = dict(environment.task_env.observation_layout)
    actor_first_layer = agent.actor.backbone[0]
    per_checkpoint = {
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "trajectory": str(trajectory_path.relative_to(ROOT)),
        "sample_count": len(records),
        "CRITIC_LOCAL_GRADIENT": {
            "fraction_Q_oracle_gt_Q_actor": float(np.mean([record["Q_oracle"] > record["Q_actor"] for record in records])),
            "fraction_directional_derivative_toward_oracle_gt_0": float(np.mean([record["directional"] > 0.0 for record in records])),
            "directional_derivative": scalar_statistics([record["directional"] for record in records]),
            "grad_action_norm": scalar_statistics(grad_action_norms),
            "finite_difference": scalar_statistics([record["finite_difference"] for record in records]),
            "finite_difference_absolute_error": scalar_statistics([
                abs(record["directional"] - record["finite_difference"]) for record in records
            ]),
        },
        "CRITIC_LANDSCAPE": {
            "counts": landscape_counts,
            "fractions": {name: count / len(records) for name, count in landscape_counts.items()},
        },
        "GRADIENT_TRANSMISSION": {
            "grad_action_norm": scalar_statistics(grad_action_norms),
            "grad_eta_norm": scalar_statistics([float(torch.linalg.vector_norm(item.grad_eta)) for item in transmission_rows]),
            "grad_u_norm": scalar_statistics([float(torch.linalg.vector_norm(item.grad_u)) for item in transmission_rows]),
            "transmission_ratio_G": scalar_statistics([float(item.ratio_G) for item in transmission_rows]),
            "transmission_ratio_tanh": scalar_statistics([float(item.ratio_tanh) for item in transmission_rows]),
            "total_transmission_ratio": scalar_statistics([float(item.ratio_total) for item in transmission_rows]),
            "per_axis": {
                "grad_action": np.mean(np.stack([item.grad_action.numpy()[0] for item in transmission_rows]), axis=0).tolist(),
                "grad_eta": np.mean(np.stack([item.grad_eta.numpy()[0] for item in transmission_rows]), axis=0).tolist(),
                "grad_u": np.mean(np.stack([item.grad_u.numpy()[0] for item in transmission_rows]), axis=0).tolist(),
                "tanh_derivative": np.mean(np.stack([item.tanh_derivative.numpy()[0] for item in transmission_rows]), axis=0).tolist(),
            },
            "eta_saturation": {
                "fraction_abs_gt_090": float(np.mean(np.abs(np.tanh(np.stack([record["mean_u"] for record in records]))) > 0.90)),
                "fraction_abs_gt_095": float(np.mean(np.abs(np.tanh(np.stack([record["mean_u"] for record in records]))) > 0.95)),
                "fraction_abs_gt_099": float(np.mean(np.abs(np.tanh(np.stack([record["mean_u"] for record in records]))) > 0.99)),
            },
        },
        "ACTOR_GRADIENT_DECOMPOSITION": gradient_decomposition,
        "GOAL_CONDITIONING": {
            "untrained_action_goal_jacobian_norm": untrained_goal_jacobian,
            "trained_mean_goal_jacobian_norm": float(np.mean([record["mean_goal_jacobian_norm"] for record in records])),
            "trained_action_goal_jacobian_norm": trained_goal_jacobian,
            "critic_goal_jacobian_norm": float(np.mean([record["critic_goal_jacobian_norm"] for record in records])),
            "Q_through_actor_goal_gradient_norm": float(np.mean([record["Q_actor_goal_gradient_norm"] for record in records])),
            "actor_to_critic_goal_jacobian_ratio": trained_goal_jacobian / (
                float(np.mean([record["critic_goal_jacobian_norm"] for record in records])) + EPSILON
            ),
            "actor_first_layer_columns": first_layer_column_statistics(actor_first_layer, actor_groups),
        },
        "CRITIC_ACTION_CONDITIONING": {
            "physical_action_per_axis": {
                "mean": np.mean(action_values, axis=0).tolist(),
                "std": np.std(action_values, axis=0).tolist(),
                "min": np.min(action_values, axis=0).tolist(),
                "max": np.max(action_values, axis=0).tolist(),
            },
            "representative_observation_std": representative_observation_std,
            "mean_action_std_to_observation_std_ratio": float(np.mean(np.std(action_values, axis=0)) / (representative_observation_std + EPSILON)),
            "critic_1_columns": critic_action_column_statistics(agent.critic_1, observation.size),
            "critic_2_columns": critic_action_column_statistics(agent.critic_2, observation.size),
            "physical_grad_action_norm": scalar_statistics(grad_action_norms),
            "hypothetical_normalized_action_gradient_norm": scalar_statistics(hypothetical_norms),
        },
    }
    frozen_data = {
        "observations": observations,
        "centers": centers,
        "generators": generators,
        "positions": positions,
        "velocities": velocities,
        "goals": goals,
        "oracle_actions": oracle_actions,
        "dt": environment.plant.config.dt,
    }
    frozen = _frozen_actor_updates(agent, frozen_data, goal_slice, seed)
    return per_checkpoint, frozen


def _aggregate(checkpoints: list[dict[str, object]], frozen: list[dict[str, object]]) -> dict[str, object]:
    directional = [entry["CRITIC_LOCAL_GRADIENT"]["fraction_directional_derivative_toward_oracle_gt_0"] for entry in checkpoints]
    q_norm = [entry["ACTOR_GRADIENT_DECOMPOSITION"]["Q_gradient_norm"] for entry in checkpoints]
    entropy_norm = [entry["ACTOR_GRADIENT_DECOMPOSITION"]["entropy_gradient_norm"] for entry in checkpoints]
    cosine = [entry["ACTOR_GRADIENT_DECOMPOSITION"]["Q_entropy_cosine"] for entry in checkpoints]
    total_transmission = [entry["GRADIENT_TRANSMISSION"]["total_transmission_ratio"]["mean"] for entry in checkpoints]
    trained_jacobian = [entry["GOAL_CONDITIONING"]["trained_action_goal_jacobian_norm"] for entry in checkpoints]
    untrained_jacobian = [entry["GOAL_CONDITIONING"]["untrained_action_goal_jacobian_norm"] for entry in checkpoints]
    critic_jacobian = [entry["GOAL_CONDITIONING"]["critic_goal_jacobian_norm"] for entry in checkpoints]
    q_only_changes = []
    current_objective_changes = []
    for result in frozen:
        for name, output in (
            ("Q_ONLY", q_only_changes),
            ("CURRENT_ACTOR_OBJECTIVE", current_objective_changes),
        ):
            values = result["trajectories"][name]
            output.append({
                "Q": values[-1]["Q_actor"] - values[0]["Q_actor"],
                "projection": values[-1]["actor_goal_projection"] - values[0]["actor_goal_projection"],
                "oracle_gap": values[-1]["oracle_gap"] - values[0]["oracle_gap"],
                "goal_jacobian": values[-1]["action_goal_jacobian_norm"] - values[0]["action_goal_jacobian_norm"],
            })
    entropy_ratios = [entropy / (q + EPSILON) for entropy, q in zip(entropy_norm, q_norm)]
    q_only_actionable = np.mean([change["Q"] > 0.0 and change["oracle_gap"] < 0.0 for change in q_only_changes]) >= 2 / 3
    q_only_goal_conditioning_decreased = np.mean(
        [change["goal_jacobian"] for change in q_only_changes]
    ) < 0.0
    current_objective_actionable = np.mean([
        change["Q"] > 0.0 and change["oracle_gap"] < 0.0
        for change in current_objective_changes
    ]) >= 2 / 3
    entropy_interaction_varies_by_seed = min(cosine) < 0.0 < max(cosine)
    goal_collapse = np.mean(trained_jacobian) < 0.25 * np.mean(untrained_jacobian) and np.mean(critic_jacobian) > 4.0 * np.mean(trained_jacobian)
    if np.mean(directional) < 0.5:
        classification = "CRITIC_LOCAL_GRADIENT_WRONG_DIRECTION"
    elif np.mean(entropy_ratios) > 10.0:
        classification = "ENTROPY_GRADIENT_DOMINATES"
    elif np.mean(cosine) < -0.8:
        classification = "Q_ENTROPY_GRADIENT_CANCELLATION"
    elif goal_collapse:
        classification = "GOAL_CONDITIONING_COLLAPSE"
    elif np.mean(total_transmission) < 0.01:
        classification = "ACTION_TO_LATENT_GRADIENT_ATTENUATION"
    elif (
        q_only_actionable
        and not current_objective_actionable
        and q_only_goal_conditioning_decreased
        and entropy_interaction_varies_by_seed
    ):
        classification = "MIXED"
    elif q_only_actionable:
        classification = "ONLINE_OPTIMIZATION_ONLY"
    else:
        classification = "MIXED"
    if classification == "ONLINE_OPTIMIZATION_ONLY" and q_only_actionable:
        gate = "PASS"
    elif classification in {"MIXED", "ACTION_TO_LATENT_GRADIENT_ATTENUATION"}:
        gate = "MARGINAL"
    else:
        gate = "FAIL"
    return {
        "fraction_directional_derivative_toward_oracle_gt_0": float(np.mean(directional)),
        "Q_gradient_norm": scalar_statistics(q_norm),
        "entropy_gradient_norm": scalar_statistics(entropy_norm),
        "entropy_to_Q_gradient_ratio": scalar_statistics(entropy_ratios),
        "Q_entropy_cosine": scalar_statistics(cosine),
        "total_transmission_ratio": scalar_statistics(total_transmission),
        "trained_action_goal_jacobian_norm": scalar_statistics(trained_jacobian),
        "untrained_action_goal_jacobian_norm": scalar_statistics(untrained_jacobian),
        "critic_goal_jacobian_norm": scalar_statistics(critic_jacobian),
        "Q_ONLY_changes": q_only_changes,
        "CURRENT_ACTOR_OBJECTIVE_changes": current_objective_changes,
        "Q_ONLY_goal_conditioning_decreased": bool(q_only_goal_conditioning_decreased),
        "entropy_interaction_varies_by_seed": bool(entropy_interaction_varies_by_seed),
        "goal_conditioning_collapse": bool(goal_collapse),
        "PRIMARY_CLASSIFICATION": classification,
        "ACTOR_GRADIENT_LEARNING_GATE": gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        nargs="+",
        default=[f"artifacts/temp_compare_physical_seed{seed}/checkpoint_latest.pt" for seed in range(3)],
    )
    parser.add_argument("--scenario", default="random_persistent_open")
    parser.add_argument("--sample-count", type=int, default=200)
    parser.add_argument("--output", default="artifacts/random_persistent/actor_gradient_learning_audit.json")
    parser.add_argument("--frozen-output", default="artifacts/random_persistent/frozen_critic_actor_only_audit.json")
    args = parser.parse_args()
    checkpoints = []
    frozen = []
    for seed, path in enumerate(args.checkpoint):
        checkpoint_result, frozen_result = _checkpoint_audit(
            ROOT / path, args.scenario, args.sample_count, seed
        )
        checkpoints.append(checkpoint_result)
        frozen.append({"checkpoint": path, **frozen_result})
    aggregate = _aggregate(checkpoints, frozen)
    result = {
        "scenario": args.scenario,
        "sample_count_per_checkpoint": args.sample_count,
        "checkpoints": checkpoints,
        "aggregate": aggregate,
        "oracle_used_for_training": False,
        "safety_semantics_changed": False,
        "synthetic_only": True,
    }
    frozen_result = {
        "scenario": args.scenario,
        "checkpoints": frozen,
        "aggregate_Q_ONLY_changes": aggregate["Q_ONLY_changes"],
        "oracle_used_for_training": False,
        "environment_training": False,
        "synthetic_only": True,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    frozen_output = ROOT / args.frozen_output
    frozen_output.parent.mkdir(parents=True, exist_ok=True)
    frozen_output.write_text(json.dumps(frozen_result, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
