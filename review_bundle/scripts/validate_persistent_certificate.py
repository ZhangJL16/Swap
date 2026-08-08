#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from math import inf
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.certified_uav import make_persistent_uav_env


def policy_authority_report(env) -> dict[str, object]:
    provider = env.certificate_provider
    failures: list[dict[str, str]] = []
    sigma_min = inf
    condition_max = 0.0
    volume_min = inf
    checked = 0
    goal_available = True
    station_available = True
    complete_recoverable = True
    neutral_center = True
    full_rank = True
    for edge_id, edge_provider in sorted(provider.providers.items()):
        provider.activate_edge(edge_id)
        provider.configure_charging_support(False)
        roots = edge_provider.root_cells[:-1] if len(edge_provider.root_cells) > 1 else edge_provider.root_cells
        edge = env.network.edges[edge_id]
        for root in roots:
            base = env.runtime._certificate_state()
            energy_low = float(root.state_bounds.energy.low)
            energy_high = float(root.state_bounds.energy.high)
            state = replace(
                base,
                position=tuple(root.reference_position),
                velocity=tuple(root.reference_velocity),
                energy=min(env.charging.config.battery_capacity, max(energy_low, 0.5 * (energy_low + energy_high))),
                explicit_task_state={"scenario": env.plant.scenario.name, "mission_phase": "TASK_RL"},
            )
            try:
                context = provider.evaluate(state, env.plant.state.timestamp)
                certificate = provider.recoverability_verifiers[edge_id].policy_authority(
                    state,
                    context,
                    env.network.nodes[edge.target].position,
                    env.plant.scenario.station_position,
                )
            except Exception as error:
                failures.append({"edge_id": edge_id, "cell_id": root.cell_id, "reason": str(error)})
                continue
            checked += 1
            sigma_min = min(sigma_min, certificate.sigma_min)
            condition_max = max(condition_max, certificate.condition_number)
            volume_min = min(volume_min, certificate.zonotope_volume)
            neutral_center &= certificate.neutral_center
            full_rank &= certificate.full_rank
            goal_available &= certificate.goal_direction_available
            station_available &= certificate.station_direction_available
            complete_recoverable &= certificate.complete_set_recoverable
            if not certificate.passed:
                failures.append({"edge_id": edge_id, "cell_id": root.cell_id, "reason": certificate.reason})
    return {
        "gate": "PASS" if checked > 0 and not failures else "FAIL",
        "neutral_center": neutral_center,
        "output_dimension": 3,
        "full_rank": full_rank,
        "minimum_sigma_min_G": None if checked == 0 else sigma_min,
        "maximum_condition_G": None if checked == 0 else condition_max,
        "minimum_zonotope_volume": None if checked == 0 else volume_min,
        "goal_direction_available": goal_available,
        "station_direction_available": station_available,
        "complete_set_recoverable": complete_recoverable,
        "number_checked": checked,
        "number_failed": len(failures),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build synthetic persistent goal-network manifests and report their gates.")
    parser.add_argument("--scenarios", nargs="+", default=["persistent_open", "persistent_obstacle", "persistent_energy_tight"])
    parser.add_argument("--output", default="artifacts/persistent/certificate_gate.json")
    args = parser.parse_args()
    results = []
    policy_results = []
    for name in args.scenarios:
        env = make_persistent_uav_env(f"{name}.json", timing_mode="functional")
        _, info = env.reset(seed=0)
        manifest = env.certificate_provider.persistent_manifest
        policy = policy_authority_report(env)
        results.append({
            "scenario": name,
            "gate": "PASS" if manifest.gate_pass else "FAIL",
            "manifest_hash": manifest.manifest_hash,
            "network_hash": manifest.goal_network_hash,
            "edge_count": len(manifest.edge_certificates),
            "recoverable_set_valid": manifest.recoverable_set_valid,
            "recoverability_action_rule_valid": manifest.recoverability_action_rule_valid,
            "complete_generator_recoverability_required": manifest.complete_generator_recoverability_required,
            "recoverable_set_version": manifest.recoverable_set_version,
            "recoverability_action_rule_version": manifest.recoverability_action_rule_version,
            "energy_field_version": manifest.energy_field_version,
            "kappa_version": manifest.kappa_version,
            "failure_reasons": manifest.failure_reasons,
            "synthetic_only": True,
        })
        policy_results.append({"scenario": name, **policy, "synthetic_only": True})
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"persistent_certificate": results, "policy_authority": policy_results}
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    persistent_pass = all(item["gate"] == "PASS" for item in results)
    policy_pass = all(item["gate"] == "PASS" for item in policy_results)
    print(f"PERSISTENT_CERTIFICATE_GATE = {'PASS' if persistent_pass else 'FAIL'}")
    print(f"POLICY_AUTHORITY_GATE = {'PASS' if policy_pass else 'FAIL'}")
    if not persistent_pass or not policy_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
