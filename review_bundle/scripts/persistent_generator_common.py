from __future__ import annotations

from typing import Any

import numpy as np

from cert_runtime.generator_sac import GeneratorTransition


def transition_from_cycle(
    observation: np.ndarray,
    next_observation: np.ndarray,
    actor_u: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    episode_id: int,
    context: dict[str, Any],
    next_context: dict[str, Any] | None,
    info: dict[str, Any],
) -> GeneratorTransition:
    """Build an immutable persistent transition from actual runtime execution."""
    telemetry = info["telemetry"]
    trace = telemetry.action_trace
    accepted = bool(info.get("accepted", False))
    epoch = str(context["certificate_epoch"])
    selected_next = {} if next_context is None else next_context
    next_epoch = epoch if next_context is None else str(selected_next["certificate_epoch"])
    candidate = None if trace.candidate is None else np.asarray(trace.candidate, dtype=np.float32)
    eta = np.tanh(np.asarray(actor_u, dtype=np.float32)) if accepted else None
    return GeneratorTransition(
        observation=np.asarray(observation, dtype=np.float32),
        next_observation=np.asarray(next_observation, dtype=np.float32),
        reward=float(reward),
        terminated=bool(terminated),
        truncated=bool(truncated),
        episode_id=int(episode_id),
        mission_phase=str(info.get("persistent_mode", "TASK_RL")),
        next_mission_phase=str(info.get("persistent_mode", "TASK_RL")),
        certificate_epoch=epoch,
        next_certificate_epoch=next_epoch,
        u=np.asarray(actor_u, dtype=np.float32) if accepted else None,
        eta=eta,
        c=None if not accepted else context.get("c"),
        G=None if not accepted else context.get("G"),
        candidate_action=candidate,
        kappa_action=np.asarray(trace.fallback, dtype=np.float32),
        executed_action=np.asarray(trace.published, dtype=np.float32),
        measured_action=np.asarray(trace.measured, dtype=np.float32),
        accepted=accepted,
        fallback_reason=info.get("fallback_reason"),
        next_c=selected_next.get("c") if selected_next.get("generator_available", False) else None,
        next_G=selected_next.get("G") if selected_next.get("generator_available", False) else None,
        next_kappa=np.asarray(selected_next.get("kappa", trace.fallback), dtype=np.float32),
        next_generator_available=bool(selected_next.get("generator_available", False)),
        next_certificate_valid=bool(selected_next.get("certificate_valid", False)),
        geometry_version=str(context.get("geometry_version", "")),
        corridor_version=str(context.get("corridor_version", "")),
        energy_version=str(context.get("energy_version", "")),
        certificate_hashes=(context.get("recovery_hash"), context.get("zonotope_hash")),
        scenario_id=str(info.get("scenario_id", "persistent")),
        certificate_manifest_hash=epoch,
        backup_triggered=bool(info.get("backup_triggered", False)),
        backup_reason=info.get("backup_reason"),
        energy=float(telemetry.state_before.energy),
        required_return_energy=float(info.get("required_return_energy", np.nan)),
        energy_margin=float(info.get("energy_margin", np.nan)),
        charging=bool(info.get("charging", False)),
        station_arrival=bool(info.get("voluntary_station_arrival", False)),
        departure_attempt=bool(info.get("departure_attempt", False)),
        departure_rejected=bool(info.get("departure_rejected", False)),
        task_id=info.get("task_id"),
        goal_id=info.get("current_goal_id"),
        tasks_completed=int(info.get("tasks_completed", 0)),
        recoverable_set_version=context.get("recoverable_set_version"),
        recoverability_action_rule_version=context.get("recoverability_action_rule_version"),
    )
