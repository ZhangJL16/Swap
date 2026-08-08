#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
from math import inf
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.certified_uav import GoalEdgeType, make_persistent_uav_env


def _root_state(env, root):
    base = env.runtime._certificate_state()
    energy_low = float(root.state_bounds.energy.low)
    energy_high = float(root.state_bounds.energy.high)
    return replace(
        base,
        position=tuple(root.reference_position),
        velocity=tuple(root.reference_velocity),
        energy=min(env.charging.config.battery_capacity, max(energy_low, 0.5 * (energy_low + energy_high))),
        explicit_task_state={"scenario": env.plant.scenario.name, "mission_phase": "TASK_RL"},
    )


def _kappa_cell_check(env, provider, chain, index, cell) -> tuple[bool, str]:
    action_valid = bool(
        np.all(np.asarray(cell.action_low) >= -env.runtime.config.a_max - 1e-12)
        and np.all(np.asarray(cell.action_high) <= env.runtime.config.a_max + 1e-12)
    )
    velocity_valid = bool(
        np.all(np.asarray(cell.state_bounds.velocity.low) >= -env.runtime.config.v_max - 1e-12)
        and np.all(np.asarray(cell.state_bounds.velocity.high) <= env.runtime.config.v_max + 1e-12)
    )
    if index + 1 < len(chain.cells):
        successor = chain.cells[index + 1]
        progress_valid = bool(
            cell.successor_target_cell == successor.cell_id
            and successor.level < cell.level
            and successor.recovery_certificate_hash in cell.dependency_hashes
        )
    else:
        progress_valid = bool(cell.level == 0 and cell.successor_target_cell is None and cell.successor_level is None)
    checks = (
        ("HASH_INVALID", cell.hash_valid),
        ("COMPLETE_SUCCESSOR_CONTAINMENT_INVALID", cell.complete_successor_containment),
        ("GEOMETRY_INVALID", cell.minimum_geometry_slack >= -1e-12),
        ("ACTUATOR_INVALID", action_valid),
        ("VELOCITY_INVALID", velocity_valid),
        ("ENERGY_E3_INVALID", cell.e3_residual >= -1e-12),
        ("STRICT_DESCENT_INVALID", progress_valid),
        (
            "ENERGY_RESERVE_INVALID",
            cell.state_bounds.energy.low
            >= cell.energy_upper + env.runtime.scenario.terminal.minimum_energy + provider.energy_reserve - 1e-12,
        ),
    )
    failure = next((name for name, valid in checks if not valid), "PASS")
    return failure == "PASS", failure


def _diagnostic_payload(edge_provider) -> dict[str, object]:
    diagnostic = edge_provider.last_generator_diagnostic
    return {
        "reason": "NO_GENERATOR_SET_RECOVERABILITY" if diagnostic is None else diagnostic.reason,
        "limiting_constraint": "NO_DIAGNOSTIC" if diagnostic is None else diagnostic.limiting_constraint,
        "largest_attempted_scale": None if diagnostic is None else diagnostic.largest_attempted_scale,
        "last_valid_scale": None if diagnostic is None else diagnostic.last_valid_scale,
        "last_invalid_scale": None if diagnostic is None else diagnostic.last_invalid_scale,
        "sigma_min_at_failure": None if diagnostic is None else diagnostic.sigma_min_at_failure,
        "volume_at_failure": None if diagnostic is None else diagnostic.volume_at_failure,
        "target_cell_id": None if diagnostic is None else diagnostic.target_cell_id,
    }


def policy_authority_report(env) -> dict[str, object]:
    provider = env.certificate_provider
    semantic_failures: list[dict[str, object]] = []
    coverage_failures: list[dict[str, object]] = []
    kappa_failures: list[dict[str, object]] = []
    no_generator_reasons: Counter[str] = Counter()
    sigma_values: list[float] = []
    condition_values: list[float] = []
    volume_values: list[float] = []
    rl_checked = 0
    rl_passed = 0
    kappa_checked = 0
    kappa_passed = 0
    neutral_center = True
    full_rank = True
    goal_available = True
    station_available = True
    complete_recoverable = True

    for edge_id, edge_provider in sorted(provider.providers.items()):
        edge = env.network.edges[edge_id]
        provider.activate_edge(edge_id)
        provider.configure_charging_support(False)
        if edge.edge_type == GoalEdgeType.RECOVERY_EDGE:
            for chain in edge_provider.manifest.chains:
                for index, cell in enumerate(chain.cells):
                    kappa_checked += 1
                    valid, reason = _kappa_cell_check(env, edge_provider, chain, index, cell)
                    kappa_passed += int(valid)
                    if not valid and len(kappa_failures) < 100:
                        kappa_failures.append({"edge_id": edge_id, "cell_id": cell.cell_id, "reason": reason})
            continue

        roots = edge_provider.root_cells[:-1] if len(edge_provider.root_cells) > 1 else edge_provider.root_cells
        for root in roots:
            rl_checked += 1
            state = _root_state(env, root)
            try:
                if edge_provider.gate_pass:
                    context = provider.evaluate(state, env.plant.state.timestamp)
                else:
                    zonotope, target = edge_provider._construct_zonotope(state, root)
                    context = None
                    if zonotope is not None and target is not None:
                        coverage_failures.append({
                            "edge_id": edge_id,
                            "cell_id": root.cell_id,
                            "reason": "EDGE_TYPED_GATE_INVALID",
                        })
                        continue
                if context is None or not context.generator_available:
                    diagnostic = _diagnostic_payload(edge_provider)
                    no_generator_reasons[str(diagnostic["reason"])] += 1
                    coverage_failures.append({"edge_id": edge_id, "cell_id": root.cell_id, **diagnostic})
                    if context is not None and not context.recovery.certified:
                        semantic_failures.append({
                            "edge_id": edge_id,
                            "cell_id": root.cell_id,
                            "reason": "NO_GENERATOR_AND_KAPPA_INVALID",
                        })
                    continue
                certificate = provider.recoverability_verifiers[edge_id].policy_authority(
                    state,
                    context,
                    env.network.nodes[edge.target].position,
                    env.plant.scenario.station_position,
                )
            except Exception as error:
                semantic_failures.append({"edge_id": edge_id, "cell_id": root.cell_id, "reason": repr(error)})
                continue
            if certificate.passed:
                rl_passed += 1
                sigma_values.append(certificate.sigma_min)
                condition_values.append(certificate.condition_number)
                volume_values.append(certificate.zonotope_volume)
            else:
                semantic_failures.append({"edge_id": edge_id, "cell_id": root.cell_id, "reason": certificate.reason})
            neutral_center &= certificate.neutral_center
            full_rank &= certificate.full_rank
            goal_available &= certificate.goal_direction_available
            station_available &= certificate.station_direction_available
            complete_recoverable &= certificate.complete_set_recoverable

    policy_pass = bool(rl_checked > 0 and not semantic_failures and not kappa_failures)
    return {
        "gate": "PASS" if policy_pass else "FAIL",
        "neutral_center": neutral_center,
        "output_dimension": 3,
        "full_rank": full_rank,
        "minimum_sigma_min_G": min(sigma_values) if sigma_values else None,
        "maximum_condition_G": max(condition_values) if condition_values else None,
        "minimum_zonotope_volume": min(volume_values) if volume_values else None,
        "goal_direction_available": goal_available,
        "station_direction_available": station_available,
        "complete_set_recoverable": complete_recoverable,
        "rl_authority_cells_checked": rl_checked,
        "rl_authority_cells_passed": rl_passed,
        "rl_authority_cells_failed": rl_checked - rl_passed,
        "rl_authority_coverage": 0.0 if rl_checked == 0 else rl_passed / rl_checked,
        "kappa_only_cells_checked": kappa_checked,
        "kappa_only_cells_passed": kappa_passed,
        "kappa_only_cells_failed": kappa_checked - kappa_passed,
        "no_generator_set_count": sum(no_generator_reasons.values()),
        "no_generator_set_reasons": dict(sorted(no_generator_reasons.items())),
        "coverage_failure_reasons": dict(sorted(Counter(
            str(item["reason"]) for item in coverage_failures
        ).items())),
        "semantic_failures": semantic_failures,
        "coverage_failures": coverage_failures[:500],
        "kappa_failures": kappa_failures,
    }


def _edge_failure_witness(env, edge_id: str) -> dict[str, object]:
    aggregate = env.certificate_provider
    provider = aggregate.providers[edge_id]
    edge = env.network.edges[edge_id]
    certificate = next(item for item in aggregate.persistent_manifest.edge_certificates if item.edge_id == edge_id)
    invalid_cell = next(
        (
            (chain, index, cell)
            for chain in provider.manifest.chains
            for index, cell in enumerate(chain.cells)
            if not (
                cell.hash_valid
                and cell.complete_successor_containment
                and cell.minimum_geometry_slack >= -1e-12
                and cell.e3_residual >= -1e-12
                and cell.state_bounds.energy.low
                >= cell.energy_upper + env.runtime.scenario.terminal.minimum_energy + provider.energy_reserve - 1e-12
            )
        ),
        None,
    )
    transition_failure_index = next(
        (index for index, valid in enumerate(provider.manifest.task_transition_verified) if not valid),
        None,
    )
    if invalid_cell is not None:
        chain, cell_index, cell = invalid_cell
    elif provider.root_cells:
        selected = min(transition_failure_index or 0, len(provider.root_cells) - 1)
        cell = provider.root_cells[selected]
        chain = provider._chains_by_id[cell.chain_id]
        cell_index = 0
    else:
        chain = cell = None
        cell_index = 0
    generator_available = False
    generator_diagnostic: dict[str, object] | None = None
    if cell is not None:
        root = chain.root
        state = _root_state(env, root)
        zonotope, _ = provider._construct_zonotope(state, root)
        generator_available = zonotope is not None
        if not generator_available:
            generator_diagnostic = _diagnostic_payload(provider)
    recovery_chain_valid = aggregate._recovery_chain_valid(provider)
    task_transition_valid = bool(
        provider.manifest.task_transition_verified
        and all(provider.manifest.task_transition_verified)
        and len(provider.manifest.task_transition_verified) == len(provider.task_reference)
    )
    exact_witness = None
    if provider.manifest.failure_witnesses:
        exact_witness = asdict(provider.manifest.failure_witnesses[0])
    elif invalid_cell is not None:
        exact_witness = {"failed_predicate": "RECOVERY_CELL_INVALID", "cell_id": cell.cell_id}
    elif transition_failure_index is not None:
        exact_witness = {"failed_predicate": "TASK_TRANSITION_INVALID", "cell_id": cell.cell_id}
    classification = "none"
    if not recovery_chain_valid:
        classification = "real_certificate_infeasibility"
    elif edge.edge_type == GoalEdgeType.RECOVERY_EDGE and not provider.gate_pass:
        classification = "typed_gate_bug"
    elif not task_transition_valid:
        classification = "real_certificate_infeasibility"
    elif not generator_available:
        classification = "constructor_limitation_or_positive_volume_infeasibility"
    action_valid = None if cell is None else bool(
        np.all(np.asarray(cell.action_low) >= -env.runtime.config.a_max - 1e-12)
        and np.all(np.asarray(cell.action_high) <= env.runtime.config.a_max + 1e-12)
    )
    velocity_valid = None if cell is None else bool(
        np.all(np.asarray(cell.state_bounds.velocity.low) >= -env.runtime.config.v_max - 1e-12)
        and np.all(np.asarray(cell.state_bounds.velocity.high) <= env.runtime.config.v_max + 1e-12)
    )
    successor = None if cell is None or cell.successor_target_cell is None else provider._cells_by_id.get(cell.successor_target_cell)
    predicate_counts: Counter[str] = Counter()
    representative_failures: dict[str, dict[str, object]] = {}
    for selected_chain in provider.manifest.chains:
        for selected_index, selected_cell in enumerate(selected_chain.cells):
            selected_action_valid = bool(
                np.all(np.asarray(selected_cell.action_low) >= -env.runtime.config.a_max - 1e-12)
                and np.all(np.asarray(selected_cell.action_high) <= env.runtime.config.a_max + 1e-12)
            )
            selected_velocity_valid = bool(
                np.all(np.asarray(selected_cell.state_bounds.velocity.low) >= -env.runtime.config.v_max - 1e-12)
                and np.all(np.asarray(selected_cell.state_bounds.velocity.high) <= env.runtime.config.v_max + 1e-12)
            )
            if selected_index + 1 < len(selected_chain.cells):
                selected_successor = selected_chain.cells[selected_index + 1]
                selected_progress = bool(
                    selected_cell.successor_target_cell == selected_successor.cell_id
                    and selected_successor.level < selected_cell.level
                )
            else:
                selected_progress = bool(selected_cell.level == 0 and selected_cell.successor_target_cell is None)
            predicates = (
                ("HASH_INVALID", selected_cell.hash_valid),
                ("COMPLETE_SUCCESSOR_CONTAINMENT_INVALID", selected_cell.complete_successor_containment),
                ("GEOMETRY_INVALID", selected_cell.minimum_geometry_slack >= -1e-12),
                ("ACTUATOR_INVALID", selected_action_valid),
                ("VELOCITY_INVALID", selected_velocity_valid),
                ("ENERGY_E3_INVALID", selected_cell.e3_residual >= -1e-12),
                ("STRICT_DESCENT_INVALID", selected_progress),
                (
                    "ENERGY_RESERVE_INVALID",
                    selected_cell.state_bounds.energy.low
                    >= selected_cell.energy_upper + env.runtime.scenario.terminal.minimum_energy + provider.energy_reserve - 1e-12,
                ),
            )
            for predicate, valid in predicates:
                if valid:
                    continue
                predicate_counts[predicate] += 1
                representative_failures.setdefault(predicate, {
                    "cell_id": selected_cell.cell_id,
                    "root_index": selected_chain.root_index,
                    "recovery_level": selected_cell.level,
                    "minimum_geometry_slack": selected_cell.minimum_geometry_slack,
                    "e3_residual": selected_cell.e3_residual,
                })
    return {
        "edge_id": edge_id,
        "edge_type": edge.edge_type.value,
        "cell_id": None if cell is None else cell.cell_id,
        "root_index": None if chain is None else chain.root_index,
        "recovery_level": None if cell is None else cell.level,
        "provider_gate_subconditions": {
            "manifest_hash_chain_valid": provider.manifest.hash_chain_valid,
            "recovery_chain_valid": recovery_chain_valid,
            "task_transition_verified": task_transition_valid,
            "typed_gate_pass": certificate.typed_gate_pass,
        },
        "hash_valid": None if cell is None else cell.hash_valid,
        "complete_successor_containment": None if cell is None else cell.complete_successor_containment,
        "minimum_geometry_slack": None if cell is None else cell.minimum_geometry_slack,
        "energy_upper": None if cell is None else cell.energy_upper,
        "successor_energy_upper": None if cell is None else cell.successor_energy_upper,
        "e3_residual": None if cell is None else cell.e3_residual,
        "state_energy_lower": None if cell is None else cell.state_bounds.energy.low,
        "required_recovery_reserve": None if cell is None else cell.energy_upper + env.runtime.scenario.terminal.minimum_energy + provider.energy_reserve,
        "task_transition_verified": task_transition_valid,
        "generator_available": generator_available,
        "generator_failure": generator_diagnostic,
        "kappa_valid": bool(
            cell is not None
            and cell.hash_valid
            and cell.complete_successor_containment
            and cell.minimum_geometry_slack >= -1e-12
            and bool(action_valid)
            and bool(velocity_valid)
            and cell.e3_residual >= -1e-12
        ),
        "kappa_progress_descent": None if cell is None else bool(
            (successor is not None and successor.level < cell.level)
            or (successor is None and cell.level == 0)
        ),
        "kappa_successor_cell": None if successor is None else successor.cell_id,
        "collision_condition": None if cell is None else cell.minimum_geometry_slack >= -1e-12,
        "velocity_condition": velocity_valid,
        "energy_successor_condition": None if cell is None else cell.e3_residual >= -1e-12,
        "exact_failure_witness": exact_witness,
        "failure_predicate_counts": dict(sorted(predicate_counts.items())),
        "representative_failures": representative_failures,
        "classification": classification,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build typed synthetic persistent goal-network manifests and report their gates.")
    parser.add_argument("--scenarios", nargs="+", default=["persistent_open", "persistent_obstacle", "persistent_energy_tight"])
    parser.add_argument("--output", default="artifacts/persistent/certificate_gate.json")
    args = parser.parse_args()
    safety_results = []
    policy_results = []
    obstacle_witnesses = []
    for name in args.scenarios:
        env = make_persistent_uav_env(f"{name}.json", timing_mode="functional")
        env.reset(seed=0)
        manifest = env.certificate_provider.persistent_manifest
        policy = policy_authority_report(env)
        safety_results.append({
            "scenario": name,
            "gate": "PASS" if manifest.gate_pass else "FAIL",
            "manifest_hash": manifest.manifest_hash,
            "network_hash": manifest.goal_network_hash,
            "edge_count": len(manifest.edge_certificates),
            "shared_bound_versions": asdict(manifest.shared_bound_versions),
            "edge_dependency_hashes": {
                item.edge_id: item.dependency_hash for item in manifest.per_edge_dependency_versions
            },
            "typed_edges": {
                item.edge_id: {
                    "edge_type": item.edge_type,
                    "recovery_chain_valid": item.recovery_chain_valid,
                    "task_transition_valid": item.task_transition_valid,
                    "rl_authority_required": item.rl_authority_required,
                    "typed_gate_pass": item.typed_gate_pass,
                }
                for item in manifest.edge_certificates
            },
            "recoverable_set_valid": manifest.recoverable_set_valid,
            "recoverability_action_rule_valid": manifest.recoverability_action_rule_valid,
            "version_consistent": manifest.version_consistent,
            "failure_reasons": manifest.failure_reasons,
            "synthetic_only": True,
        })
        policy_results.append({"scenario": name, **policy, "synthetic_only": True})
        if name == "persistent_obstacle":
            for edge_id in ("recover_C_S", "task_C_B", "task_C_D", "task_D_C"):
                obstacle_witnesses.append(_edge_failure_witness(env, edge_id))
    payload = {
        "persistent_safety_gate": safety_results,
        "policy_authority_gate": policy_results,
        "policy_authority_coverage": [
            {
                "scenario": item["scenario"],
                "checked": item["rl_authority_cells_checked"],
                "passed": item["rl_authority_cells_passed"],
                "failed": item["rl_authority_cells_failed"],
                "coverage": item["rl_authority_coverage"],
                "no_generator_set_reasons": item["no_generator_set_reasons"],
            }
            for item in policy_results
        ],
        "persistent_obstacle_failure_witnesses": obstacle_witnesses,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    safety_pass = all(item["gate"] == "PASS" for item in safety_results)
    policy_pass = all(item["gate"] == "PASS" for item in policy_results)
    print(f"PERSISTENT_SAFETY_GATE = {'PASS' if safety_pass else 'FAIL'}")
    print(f"PERSISTENT_CERTIFICATE_GATE = {'PASS' if safety_pass else 'FAIL'}")
    print(f"POLICY_AUTHORITY_GATE = {'PASS' if policy_pass else 'FAIL'}")
    if not safety_pass or not policy_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
