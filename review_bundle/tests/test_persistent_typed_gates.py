from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
import unittest

import numpy as np

from cert_runtime.certificates import StateCellBounds
from cert_runtime.interval import Interval
from cert_runtime.persistent_authority import ExecutionAuthority, PersistentAuthorityInput, PersistentExecutionAuthority
from cert_runtime.types import Interval3
from envs.certified_uav import (
    EdgeDependencyBinding,
    GoalEdgeType,
    PersistentGoalEdgeCertificate,
    SharedBoundVersions,
    edge_dependency_bindings_valid,
    shared_bound_versions_consistent,
    typed_edge_gate_pass,
)
from envs.certified_uav.mission_certificate import GeneratorConstructionDiagnostic
from scripts.validate_persistent_certificate import _diagnostic_payload, policy_authority_report


@dataclass
class _BaseState:
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    energy: float = 3.0
    explicit_task_state: dict[str, str] | None = None


def shared_versions(**changes) -> SharedBoundVersions:
    values = {
        "dynamics_version": "dynamics-v1",
        "dynamics_hash": "dynamics-hash",
        "tracking_version": "tracking-v1",
        "tracking_hash": "tracking-hash",
        "energy_version": "energy-v1",
        "energy_hash": "energy-hash",
        "terminal_version": "terminal-v1",
        "terminal_hash": "terminal-hash",
        "recoverable_set_version": "recoverable-v1",
        "recoverability_action_rule_version": "action-rule-v1",
        "runtime_configuration_version": "runtime-v1",
    }
    values.update(changes)
    return SharedBoundVersions(**values)


def binding(edge_id: str, geometry: str, kappa: str) -> EdgeDependencyBinding:
    provisional = EdgeDependencyBinding(
        edge_id=edge_id,
        edge_type=GoalEdgeType.RECOVERY_EDGE.value,
        scenario_id=f"scenario:{edge_id}",
        geometry_version=geometry,
        geometry_certificate_hash=f"geometry-hash:{edge_id}",
        kappa_version=kappa,
        kappa_certificate_hash=f"kappa-hash:{edge_id}",
        corridor_certificate_hash=f"corridor-hash:{edge_id}",
        mission_manifest_hash=f"manifest:{edge_id}",
        dependency_hash="pending",
    )
    return replace(provisional, dependency_hash=provisional.expected_hash)


def edge_certificate(item: EdgeDependencyBinding) -> PersistentGoalEdgeCertificate:
    provisional = PersistentGoalEdgeCertificate(
        edge_id=item.edge_id,
        source="A",
        target="S",
        edge_type=item.edge_type,
        recovery_manifest_hash=item.mission_manifest_hash,
        recovery_gate_pass=True,
        typed_gate_pass=True,
        recovery_chain_valid=True,
        task_transition_valid=False,
        rl_authority_required=False,
        complete_successor_support=True,
        energy_upper=1.0,
        dependency_hashes=(item.mission_manifest_hash, "network", item.dependency_hash),
        dependency_binding_hash=item.dependency_hash,
        certificate_hash="pending",
    )
    return replace(provisional, certificate_hash=provisional.expected_hash)


class PersistentVersionSemanticsTests(unittest.TestCase):
    def test_edge_local_versions_may_differ(self):
        first = binding("recover_A_S", "geometry-A", "kappa-A")
        second = binding("recover_B_S", "geometry-B", "kappa-B")
        self.assertTrue(edge_dependency_bindings_valid(
            (first, second),
            (edge_certificate(first), edge_certificate(second)),
        ))

    def test_shared_bound_version_mismatch_fails(self):
        baseline = shared_versions()
        for field, value in (
            ("dynamics_version", "dynamics-v2"),
            ("tracking_version", "tracking-v2"),
            ("energy_version", "energy-v2"),
        ):
            with self.subTest(field=field):
                self.assertFalse(shared_bound_versions_consistent((baseline, replace(baseline, **{field: value}))))

    def test_wrong_edge_kappa_hash_fails(self):
        valid = binding("recover_A_S", "geometry-A", "kappa-A")
        tampered = replace(valid, kappa_certificate_hash="wrong-kappa-hash")
        self.assertFalse(edge_dependency_bindings_valid((tampered,), (edge_certificate(valid),)))

    def test_wrong_edge_manifest_hash_fails(self):
        valid = binding("recover_A_S", "geometry-A", "kappa-A")
        tampered = replace(valid, mission_manifest_hash="wrong-manifest")
        self.assertFalse(edge_dependency_bindings_valid((tampered,), (edge_certificate(valid),)))


class TypedEdgeSemanticsTests(unittest.TestCase):
    def test_recovery_edge_does_not_require_generator(self):
        self.assertTrue(typed_edge_gate_pass(
            GoalEdgeType.RECOVERY_EDGE,
            recovery_chain_valid=True,
            task_transition_valid=False,
        ))

    def test_recovery_edge_requires_valid_kappa(self):
        self.assertFalse(typed_edge_gate_pass(
            GoalEdgeType.RECOVERY_EDGE,
            recovery_chain_valid=False,
            task_transition_valid=True,
        ))

    def test_task_edge_requires_recoverable_generator_when_rl_authority_claimed(self):
        self.assertFalse(typed_edge_gate_pass(
            GoalEdgeType.TASK_EDGE,
            recovery_chain_valid=True,
            task_transition_valid=False,
        ))

    def test_departure_edge_requires_recoverable_generator(self):
        self.assertFalse(typed_edge_gate_pass(
            GoalEdgeType.DEPARTURE_EDGE,
            recovery_chain_valid=True,
            task_transition_valid=False,
        ))

    def test_no_generator_with_valid_kappa_is_backup_not_safety_failure(self):
        decision = PersistentExecutionAuthority.evaluate(PersistentAuthorityInput(
            persistent_mode="TASK_RL",
            energy_margin=3.0,
            backup_switch_margin=1.0,
            persistent_certificate_valid=True,
            certificate_valid=True,
            kappa_valid=True,
            generator_available=False,
            recoverable_set_member=True,
            recoverability_action_verified=False,
            policy_authority_pass=False,
            charging_state=False,
            departure_allowed=True,
            charging_support_verified=False,
            station_hold_valid=False,
        ))
        self.assertEqual(decision.authority, ExecutionAuthority.KAPPA_BACKUP)
        self.assertEqual(decision.reason, "NO_GENERATOR_SET")


class PolicyAuthorityAccountingTests(unittest.TestCase):
    def test_policy_authority_metrics_exclude_kappa_only_cells(self):
        bounds = StateCellBounds(
            Interval3((-0.1,) * 3, (0.1,) * 3),
            Interval3((-0.1,) * 3, (0.1,) * 3),
            Interval(2.0, 3.0),
        )
        recovery_cell = SimpleNamespace(
            cell_id="recovery-cell",
            level=0,
            successor_target_cell=None,
            successor_level=None,
            dependency_hashes=(),
            hash_valid=True,
            complete_successor_containment=True,
            minimum_geometry_slack=0.1,
            action_low=(-0.05,) * 3,
            action_high=(0.05,) * 3,
            state_bounds=bounds,
            e3_residual=0.0,
            energy_upper=0.0,
        )
        task_root = SimpleNamespace(
            cell_id="task-root",
            reference_position=(0.0, 0.0, 0.0),
            reference_velocity=(0.0, 0.0, 0.0),
            state_bounds=bounds,
        )
        recovery_provider = SimpleNamespace(
            manifest=SimpleNamespace(chains=(SimpleNamespace(cells=(recovery_cell,)),)),
            energy_reserve=0.0,
        )
        task_provider = SimpleNamespace(root_cells=(task_root,), gate_pass=True)
        policy_certificate = SimpleNamespace(
            passed=True,
            sigma_min=0.05,
            condition_number=1.0,
            zonotope_volume=0.001,
            neutral_center=True,
            full_rank=True,
            goal_direction_available=True,
            station_direction_available=True,
            complete_set_recoverable=True,
            reason="PASS",
        )
        parent = SimpleNamespace(
            providers={"recover": recovery_provider, "task": task_provider},
            activate_edge=lambda edge_id: None,
            configure_charging_support=lambda required: None,
            evaluate=lambda state, timestamp: SimpleNamespace(generator_available=True),
            recoverability_verifiers={"task": SimpleNamespace(policy_authority=lambda *args: policy_certificate)},
        )
        edges = {
            "recover": SimpleNamespace(edge_type=GoalEdgeType.RECOVERY_EDGE, target="S"),
            "task": SimpleNamespace(edge_type=GoalEdgeType.TASK_EDGE, target="B"),
        }
        env = SimpleNamespace(
            certificate_provider=parent,
            runtime=SimpleNamespace(
                config=SimpleNamespace(a_max=np.ones(3), v_max=np.ones(3)),
                scenario=SimpleNamespace(terminal=SimpleNamespace(minimum_energy=0.0)),
                _certificate_state=lambda: _BaseState(),
            ),
            charging=SimpleNamespace(config=SimpleNamespace(battery_capacity=30.0)),
            plant=SimpleNamespace(
                state=SimpleNamespace(timestamp=0.0),
                scenario=SimpleNamespace(name="fixture", station_position=np.zeros(3)),
            ),
            network=SimpleNamespace(edges=edges, nodes={"B": SimpleNamespace(position=np.ones(3))}),
        )
        report = policy_authority_report(env)
        self.assertEqual(report["kappa_only_cells_checked"], 1)
        self.assertEqual(report["rl_authority_cells_checked"], 1)
        self.assertEqual(report["minimum_sigma_min_G"], 0.05)
        self.assertTrue(report["station_direction_available"])

    def test_obstacle_failure_witness_reports_limiting_constraint(self):
        provider = SimpleNamespace(last_generator_diagnostic=GeneratorConstructionDiagnostic(
            "NO_GENERATOR_SET_COLLISION",
            "TARGET_RECOVERY_GEOMETRY",
            0.02,
            None,
            0.0,
            0.005,
            1e-6,
            "cell-1",
        ))
        payload = _diagnostic_payload(provider)
        self.assertEqual(payload["reason"], "NO_GENERATOR_SET_COLLISION")
        self.assertEqual(payload["limiting_constraint"], "TARGET_RECOVERY_GEOMETRY")


if __name__ == "__main__":
    unittest.main()
