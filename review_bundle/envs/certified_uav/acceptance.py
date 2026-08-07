from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
from json import dumps
from typing import Any

import numpy as np

from cert_runtime import SimulatedWatchdog, WCETContract

from . import CertifiedUAVConfig, make_certified_uav_env


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def certificate_state_digest(snapshot: Any) -> str:
    return sha256(dumps(_jsonable(snapshot), sort_keys=True).encode()).hexdigest()


def run_acceptance_cycle(
    scenario_id: str,
    seed: int,
    actor_output: np.ndarray | None = None,
    *,
    freeze_certificate_epoch: bool = True,
    timing_mode: str = "wall_clock",
) -> tuple[dict[str, Any], Any]:
    scenario_file = scenario_id if scenario_id.endswith(".json") else f"{scenario_id}.json"
    if scenario_id in {
        "actor_nonfinite",
        "stale_certificate",
        "watchdog_deadline",
        "energy_contract_mismatch",
        "corridor_hash_tamper",
    }:
        scenario_file = "open_corridor.json"
    config = None
    if scenario_id == "insufficient_sensing":
        config = CertifiedUAVConfig(lidar_range=0.01)
        scenario_file = "open_corridor.json"
    runtime = make_certified_uav_env(
        scenario_file,
        config,
        freeze_certificate_epoch=freeze_certificate_epoch,
        timing_mode=timing_mode,
    )
    observation, reset_info = runtime.reset(seed=seed)
    before_calls = runtime.actor.calls
    requested = np.array([0.2, -0.3, 0.1]) if actor_output is None else np.asarray(actor_output, dtype=np.float64)
    if scenario_id == "actor_nonfinite":
        requested = np.array([np.nan, 0.0, 0.0])
    elif scenario_id == "stale_certificate":
        original = runtime.actor.sample_u

        def mutate(observation_value):
            runtime.geometry.version += 1
            return original(observation_value)

        runtime.actor.sample_u = mutate
    elif scenario_id == "watchdog_deadline":
        runtime.watchdog = SimulatedWatchdog(0.0, WCETContract(control_period_seconds=runtime.config.dt))
    elif scenario_id == "energy_contract_mismatch":
        object.__setattr__(runtime.calibration.energy, "version", "tampered-energy-version")
        runtime._frozen_preparation = None
    elif scenario_id == "corridor_hash_tamper":
        original_prepare = runtime.prepare_certificate_cycle

        def tampered_prepare():
            preparation = original_prepare()
            closure = preparation.closure_result
            if closure is not None and closure.zonotope_certificate is not None:
                bad_zonotope = replace(closure.zonotope_certificate, complete_set_inclusion_hash=None)
                closure = replace(closure, zonotope_certificate=bad_zonotope)
                preparation = replace(preparation, closure_result=closure)
            return preparation

        runtime.prepare_certificate_cycle = tampered_prepare
    state_before = runtime.plant.state.copy()
    next_observation, reward, terminated, truncated, info = runtime.step(requested)
    telemetry = info["telemetry"]
    trace = telemetry.action_trace
    preparation = runtime.last_preparation
    closure = preparation.closure_result if preparation is not None else None
    zonotope_certificate = closure.zonotope_certificate if closure is not None else None
    zonotope = zonotope_certificate.zonotope if zonotope_certificate is not None else None
    record = runtime.replay.records[-1]
    manifest = info.get("certificate_manifest")
    recovery_hash = record.recovery_certificate_hash
    energy_hash = None
    if manifest is not None:
        energy_entries = [entry.proof_hash for entry in manifest.entries if entry.object_type == "recovery-energy"]
        energy_hash = sha256("".join(sorted(energy_entries)).encode()).hexdigest() if energy_entries else None
    epoch = info["certificate_epoch"]
    output = {
        "cycle_index": 0,
        "scenario_id": scenario_id,
        "reset_certificate_ready": reset_info.get("certificate_ready"),
        "state_before": _jsonable(state_before),
        "task_observation_shape": list(observation.shape),
        "certificate_state_digest": certificate_state_digest(record.certificate_state),
        "certificate_epoch": epoch.epoch_id,
        "geometry_version": telemetry.geometry_version,
        "corridor_version": telemetry.corridor_version,
        "recovery_certificate_hash": recovery_hash,
        "energy_certificate_hash": energy_hash,
        "zonotope_certificate_hash": record.inclusion_certificate_hash,
        "kappa": list(record.recovery_action),
        "c": None if zonotope is None else list(zonotope.center),
        "G": None if zonotope is None else [list(row) for row in zonotope.generators],
        "sigma_min": None if zonotope is None else zonotope.sigma_min_lower_bound,
        "det_G": None if zonotope is None else zonotope.determinant,
        "zonotope_volume": None if zonotope is None else 8.0 * abs(zonotope.determinant),
        "condition_number": None if zonotope is None else zonotope.condition_number_upper_bound,
        "actor_called": runtime.actor.calls > before_calls,
        "u": None if record.nominal_pre_squash_u is None else list(record.nominal_pre_squash_u),
        "eta": None if record.squashed_eta is None else list(record.squashed_eta),
        "a_candidate": None if record.candidate_action is None else list(record.candidate_action),
        "accepted": record.accepted,
        "fallback_reason": record.fallback_reason,
        "a_exec": list(record.executed_action),
        "a_measured": None if record.measured_tracking_action is None else list(record.measured_tracking_action),
        "state_after": _jsonable(telemetry.state_after),
        "energy_before": telemetry.state_before.energy,
        "energy_cost": telemetry.energy_cost,
        "energy_after": telemetry.state_after.energy,
        "collision": telemetry.collision,
        "terminal_admissible": telemetry.terminal_admissible,
        "terminated": terminated,
        "truncated": truncated,
        "reward": reward,
        "deadline": runtime.watchdog.deadline_seconds,
        "elapsed_time": runtime.last_stage_timings.get("T_total", 0.0),
        "stage_timings": runtime.last_stage_timings,
        "published_once": info["publication_count"] == 1,
        "plant_input_matches_exec": np.array_equal(trace.published, np.asarray(record.executed_action)),
        "manifest_integrity": bool(manifest is not None and manifest.verify_integrity()),
    }
    if output["accepted"]:
        expected = np.asarray(output["c"]) + np.asarray(output["G"]) @ np.tanh(np.asarray(output["u"]))
        if not np.allclose(expected, output["a_candidate"], atol=1e-12):
            raise AssertionError("accepted affine-tanh mapping mismatch")
        if not np.array_equal(np.asarray(output["a_exec"]), np.asarray(output["a_candidate"])):
            raise AssertionError("accepted action was not executed")
    elif not np.array_equal(np.asarray(output["a_exec"]), np.asarray(output["kappa"])):
        raise AssertionError("fallback did not execute kappa")
    if not output["plant_input_matches_exec"] or not output["published_once"]:
        raise AssertionError("execution publication invariant failed")
    if not np.isclose(output["energy_after"], output["energy_before"] - output["energy_cost"]):
        raise AssertionError("energy accounting mismatch")
    return output, runtime


SCENARIO_EXPECTATIONS = {
    "open_corridor": None,
    "narrow_corridor": "NO_GENERATOR_SET",
    "invalid_corridor": "INITIAL_STATE_OUTSIDE_CORRIDOR_SUFFIX",
    "insufficient_energy": "INSUFFICIENT_RECOVERY_RESERVE",
    "actor_nonfinite": "CERTIFIER_EXCEPTION",
    "stale_certificate": "CERTIFIER_EXCEPTION",
    "watchdog_deadline": "WATCHDOG_DEADLINE",
    "insufficient_sensing": "INSUFFICIENT_SENSING_FOR_BRAKING_TUBE",
    "energy_contract_mismatch": "EnergyCalibrationContract-expired-or-out-of-domain",
    "corridor_hash_tamper": "STALE_OR_INCOMPLETE_BUNDLE",
}


def scenario_matrix(seed: int = 0) -> list[dict[str, Any]]:
    rows = []
    for scenario_id, expected in SCENARIO_EXPECTATIONS.items():
        trace, _ = run_acceptance_cycle(scenario_id, seed)
        actual = trace["fallback_reason"]
        passed = trace["accepted"] if scenario_id == "open_corridor" else (not trace["accepted"] and actual == expected)
        rows.append(
            {
                "scenario": scenario_id,
                "certificate_valid": trace["reset_certificate_ready"],
                "actor_called": trace["actor_called"],
                "generator_available": trace["G"] is not None,
                "accepted": trace["accepted"],
                "executed_action": trace["a_exec"],
                "expected_reason": expected,
                "actual_reason": actual,
                "pass": passed,
            }
        )
    return rows
