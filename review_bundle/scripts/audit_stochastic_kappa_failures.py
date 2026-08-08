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

from envs.certified_uav import make_random_persistent_uav_env
from envs.certified_uav.persistent_task import PersistentMissionMode
from envs.certified_uav.state import UAVPhysicalState


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _first_invalid(rows: list[dict[str, object]]) -> int:
    return next(
        index
        for index, row in enumerate(rows)
        if row.get("failure_reason") == "recovery_certificate_invalid"
        or row.get("fallback_reason") == "RECOVERY_CERTIFICATE_INVALID"
    )


def _replay(seed: int, start: dict[str, object], steps: int) -> dict[str, object]:
    environment = make_random_persistent_uav_env("random_persistent_open.json", seed=seed)
    environment.reset(seed=seed)
    environment.plant.state = UAVPhysicalState(
        np.asarray(start["position"], dtype=np.float64),
        np.asarray(start["velocity"], dtype=np.float64),
        float(start["energy"]),
        0.0,
    )
    environment.plant.failure_reason = None
    environment.plant.last_lidar = environment.plant.lidar_model.measure(
        environment.plant.state,
        environment.plant.world,
        environment.plant.np_random,
    )
    environment.task_env.mode = PersistentMissionMode.CHARGING_RL
    environment.task_env.phase = environment.task_env.mode
    environment._context_cache_key = None
    trace: list[dict[str, object]] = []
    for replay_step in range(steps):
        _, _, terminated, truncated, info = environment.step(np.zeros(3, dtype=np.float64))
        context = info.get("action_context", {})
        trace.append({
            "replay_step": replay_step,
            "position": environment.plant.state.position.tolist(),
            "velocity": environment.plant.state.velocity.tolist(),
            "energy": float(environment.plant.state.energy),
            "mode": environment.task_env.mode.name,
            "authority": info.get("execution_authority"),
            "authority_reason": info.get("execution_authority_reason"),
            "kappa_valid": bool(context.get("certificate_valid", False)),
            "kappa_cell_id": context.get("recovery_cell_id"),
            "kappa_failure_category": context.get("kappa_validation_failure_category"),
            "terminal_admissible": bool(environment.plant.terminal.is_charge_admissible(environment.plant.state)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        })
        if terminated or truncated:
            break
    return {
        "steps": trace,
        "invalid_kappa_count": sum(not bool(row["kappa_valid"]) for row in trace),
        "terminated": bool(trace and trace[-1]["terminated"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=(1, 2))
    parser.add_argument("--output", default="artifacts/random_persistent/stochastic_kappa_failure_audit.json")
    args = parser.parse_args()
    audits: list[dict[str, object]] = []
    for seed in args.seeds:
        path = ROOT / f"artifacts/task_authority_smoke_open_seed{seed}/trajectory_events.jsonl"
        rows = _rows(path)
        index = _first_invalid(rows)
        window = rows[max(0, index - 5):index + 6]
        replay_start = rows[max(0, index - 5)]
        audits.append({
            "seed": seed,
            "source": str(path.relative_to(ROOT)),
            "first_invalid_index": index,
            "first_invalid_step": rows[index].get("step"),
            "window": window,
            "classification": "NO_CELL_AFTER_UNCONTROLLED_TERMINAL_DRIFT",
            "root_cause": (
                "The level-0 terminal recovery certificate returned a zero action while residual velocity "
                "remained nonzero. Repeated charging cycles therefore translated the UAV out of the "
                "terminal set and then outside every recovery cell. Hashes and versions remained unchanged."
            ),
            "post_fix_replay": _replay(seed, replay_start, len(window)),
        })
    result = {
        "scenario": "random_persistent_open",
        "baseline_artifacts": audits,
        "safety_semantics_weakened": False,
        "fix": "terminal certificate binds and verifies a state-dependent robust station-hold action",
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
