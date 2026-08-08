from __future__ import annotations

from types import SimpleNamespace
import unittest
from dataclasses import fields

import numpy as np

from cert_runtime.generator_sac import GeneratorTransition, PersistentGeneratorSAC
from cert_runtime.types import Interval3, Zonotope3
from envs.certified_uav import PersistentGoalCertificateManifest, PersistentMissionMode, make_persistent_uav_env
from envs.certified_uav.recoverability import RecoverabilityVerifier


class _EnvelopeBuilder:
    def __init__(self, energy_low: float = 8.0) -> None:
        self.energy_low = energy_low
        self.dynamics = SimpleNamespace(version="dynamics-v1")
        self.energy = SimpleNamespace(version="energy-v1")

    def propagate_zonotope(self, state, action_set):
        del state, action_set
        return SimpleNamespace(
            energy_low=self.energy_low,
            velocity=Interval3((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1)),
            dynamics_bound_version="dynamics-v1",
            energy_bound_version="energy-v1",
        )

    def propagate_point_action(self, state, action):
        del state, action
        return SimpleNamespace(
            position=Interval3((0.0, 0.0, 0.0), (0.1, 0.1, 0.1)),
            velocity=Interval3((-0.01, -0.01, -0.01), (0.01, 0.01, 0.01)),
        )


class _Provider:
    energy_reserve = 1.0

    def __init__(self) -> None:
        self.target = SimpleNamespace(
            cell_id="lower-cell",
            energy_upper=5.0,
            hash_valid=True,
            complete_successor_containment=True,
            minimum_geometry_slack=0.2,
            state_bounds=SimpleNamespace(
                velocity=Interval3((-0.2, -0.2, -0.2), (0.2, 0.2, 0.2)),
            ),
            dynamics_version="dynamics-v1",
            tracking_version="tracking-v1",
            energy_version="energy-v1",
            terminal_version="terminal-v1",
            kappa_version="kappa-v1",
            recovery_certificate_hash="recovery-hash",
        )
        self.geometry_ok = True

    def _target_root(self, root_index):
        return self.target if root_index is not None else None

    def _candidate_successor_in_root(self, state, action_set, target):
        del state, action_set, target
        return self.geometry_ok

    @staticmethod
    def _versions():
        return ("geometry-v1", "dynamics-v1", "tracking-v1", "energy-v1", "terminal-v1", "kappa-v1")


def _fixture(energy_low: float = 8.0):
    provider = _Provider()
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            a_max=np.ones(3),
            minimum_generator_sigma=0.05,
        ),
        calibration=SimpleNamespace(
            dynamics=SimpleNamespace(version="dynamics-v1"),
            tracking=SimpleNamespace(version="tracking-v1"),
            energy=SimpleNamespace(version="energy-v1"),
            terminal=SimpleNamespace(version="terminal-v1"),
        ),
        envelope_builder=_EnvelopeBuilder(energy_low),
        scenario=SimpleNamespace(
            terminal=SimpleNamespace(
                minimum_energy=1.0,
                position_low=np.zeros(3),
                position_high=np.ones(3),
                velocity_abs_max=np.ones(3) * 0.1,
                is_charge_admissible=lambda state: True,
            ),
        ),
        plant=SimpleNamespace(state=SimpleNamespace()),
    )
    state = SimpleNamespace(
        position=(0.0, 0.0, 0.0),
        bound_versions={
            "dynamics": "dynamics-v1",
            "tracking": "tracking-v1",
            "energy": "energy-v1",
            "terminal": "terminal-v1",
        },
        snapshot=lambda: {"state": "fixture"},
    )
    context = SimpleNamespace(
        recovery=SimpleNamespace(certified=True, certificate_hash="recovery-hash"),
        required_energy=7.0,
        current_energy_margin=2.0,
        recovery_cell_id="current-cell",
        root_index=1,
        closure=SimpleNamespace(
            zonotope_certificate=SimpleNamespace(
                zonotope=Zonotope3.diagonal(np.zeros(3), (0.1, 0.1, 0.1)),
            ),
        ),
    )
    return RecoverabilityVerifier(runtime, provider), runtime, provider, state, context


class RecoverabilitySemanticsTests(unittest.TestCase):
    def test_recoverable_set_membership(self):
        verifier, _, _, state, context = _fixture()
        certificate = verifier.membership(state, context)
        self.assertTrue(certificate.recoverable)
        context.current_energy_margin = -0.01
        self.assertFalse(verifier.membership(state, context).recoverable)

    def test_recoverable_action_successor_inclusion(self):
        verifier, _, _, state, context = _fixture()
        result = verifier.certify_action_set(state, context.closure.zonotope_certificate.zonotope, context)
        self.assertTrue(result.verified)
        self.assertTrue(result.collision_successor_inclusion)
        self.assertTrue(result.energy_successor_inclusion)

    def test_energy_successor_uses_next_recovery_cost(self):
        verifier, _, _, state, context = _fixture(energy_low=6.9)
        self.assertGreater(context.current_energy_margin, 0.0)
        result = verifier.certify_action_set(state, context.closure.zonotope_certificate.zonotope, context)
        self.assertFalse(result.verified)
        self.assertFalse(result.energy_successor_inclusion)
        self.assertEqual(result.successor_required_energy, 7.0)

    def test_collision_and_energy_jointly_enter_a_rec(self):
        verifier, _, provider, state, context = _fixture()
        provider.geometry_ok = False
        collision_failure = verifier.certify_action_set(state, context.closure.zonotope_certificate.zonotope, context)
        self.assertFalse(collision_failure.verified)
        self.assertFalse(collision_failure.collision_successor_inclusion)
        provider.geometry_ok = True
        verifier.runtime.envelope_builder.energy_low = 6.0
        energy_failure = verifier.certify_action_set(state, context.closure.zonotope_certificate.zonotope, context)
        self.assertFalse(energy_failure.verified)
        self.assertFalse(energy_failure.energy_successor_inclusion)

    def test_recursive_recoverability_two_steps(self):
        verifier, _, _, state, context = _fixture()
        action_set = context.closure.zonotope_certificate.zonotope
        first = verifier.certify_action_set(state, action_set, context)
        second = verifier.certify_action_set(state, action_set, context)
        self.assertTrue(first.verified and second.verified)

    def test_generator_complete_set_recoverability_and_policy_authority(self):
        verifier, _, _, state, context = _fixture()
        result = verifier.policy_authority(state, context, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        self.assertTrue(result.complete_set_recoverable)
        self.assertTrue(result.passed)
        self.assertTrue(result.goal_direction_available)
        self.assertTrue(result.station_direction_available)

    def test_continuous_departure_attempt_is_detected(self):
        verifier, runtime, _, state, _ = _fixture()
        self.assertTrue(verifier.successor_stays_in_charging_set(state, np.zeros(3)))
        runtime.envelope_builder.propagate_point_action = lambda state, action: SimpleNamespace(
            position=Interval3((0.9, 0.0, 0.0), (1.1, 0.1, 0.1)),
            velocity=Interval3((-0.01, -0.01, -0.01), (0.01, 0.01, 0.01)),
        )
        self.assertFalse(verifier.successor_stays_in_charging_set(state, np.array([0.1, 0.0, 0.0])))


class SinglePolicyAuthorityTests(unittest.TestCase):
    def test_main_path_has_one_three_dimensional_policy_and_neutral_center(self):
        environment = make_persistent_uav_env("persistent_open.json")
        self.assertEqual(environment.trainable_policy_count, 1)
        self.assertEqual(environment.action_space.shape, (3,))
        self.assertEqual(environment.runtime.generator_center_mode, "safety_neutral")
        self.assertFalse(hasattr(environment, "energy_management_policy"))
        agent = PersistentGeneratorSAC(environment.observation_space.shape[0])
        self.assertEqual(agent.action_dimension, 3)

    def test_voluntary_station_approach_retains_rl_authority(self):
        environment = make_persistent_uav_env("persistent_open.json")
        environment.task_env.reset(seed=0)
        task_id = environment.task_env.manager.current_task.task_id
        environment.task_env.request_return(forced=False)
        self.assertEqual(environment.task_env.mode, PersistentMissionMode.TASK_RL)
        self.assertTrue(environment.task_env.voluntary_station_approach)
        self.assertEqual(environment.task_env.manager.current_task.task_id, task_id)

    def test_backup_boundaries_switch_authority_to_kappa(self):
        environment = make_persistent_uav_env("persistent_open.json")
        environment.certificate_provider = SimpleNamespace(gate_pass=True)
        environment.policy_authority_certificate = SimpleNamespace(passed=True)
        valid = {
            "certificate_valid": True,
            "recoverable_set_member": True,
            "generator_available": True,
            "recoverability_action_verified": True,
        }
        environment.task_env.energy_margin = environment.charging.config.forced_return_margin
        self.assertEqual(environment._backup_reason(valid), "ENERGY_MARGIN_BACKUP_SWITCH")
        environment._begin_backup("ENERGY_MARGIN_BACKUP_SWITCH")
        self.assertEqual(environment.task_env.mode, PersistentMissionMode.BACKUP_RECOVERY)

    def test_no_generator_and_certificate_failure_switch_to_backup(self):
        environment = make_persistent_uav_env("persistent_open.json")
        environment.certificate_provider = SimpleNamespace(gate_pass=True)
        environment.policy_authority_certificate = SimpleNamespace(passed=True)
        environment.task_env.energy_margin = environment.charging.config.forced_return_margin + 1.0
        no_generator = {
            "certificate_valid": True,
            "recoverable_set_member": True,
            "generator_available": False,
            "generator_status": "NO_GENERATOR_SET",
            "recoverability_action_verified": None,
        }
        self.assertEqual(environment._backup_reason(no_generator), "NO_GENERATOR_SET")
        invalid = dict(no_generator, certificate_valid=False, failure_reason="VERSION_MISMATCH")
        self.assertEqual(environment._backup_reason(invalid), "VERSION_MISMATCH")

    def test_policy_output_shape_is_strict(self):
        environment = make_persistent_uav_env("persistent_open.json")
        with self.assertRaisesRegex(ValueError, r"shape \(3,\)"):
            environment.step(np.zeros(4))

    def test_rl_dwell_in_charge_set_applies_synthetic_charging(self):
        environment = make_persistent_uav_env("persistent_open.json")
        environment.task_env.reset(seed=0)
        environment.plant.state.position = environment.plant.scenario.station_position.copy()
        environment.plant.state.velocity = np.zeros(3)
        environment.plant.state.energy = 5.0
        _, _, _, _, info = environment.task_env.step(np.zeros(3))
        result = environment.charging.apply_during_motion_cycle(environment.plant, info["telemetry"])
        self.assertAlmostEqual(result.charged_energy, 0.4)
        self.assertAlmostEqual(environment.plant.state.energy, 5.0 - info["telemetry"].energy_cost + 0.4)

    def test_manifest_binds_recoverability_and_physical_versions(self):
        names = {item.name for item in fields(PersistentGoalCertificateManifest)}
        self.assertTrue({
            "recoverable_set_version",
            "recoverability_action_rule_version",
            "energy_field_version",
            "kappa_version",
            "geometry_version",
            "tracking_version",
            "dynamics_version",
        }.issubset(names))

    def test_persistent_replay_records_actual_authority_metadata_without_aliasing(self):
        observation = np.zeros(4, dtype=np.float32)
        executed = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        transition = GeneratorTransition(
            observation, observation, 0.0, False, False, 1, "TASK_RL", "TASK_RL",
            "manifest", "manifest", None, None, None, None, None,
            np.zeros(3), executed, executed, False, "NO_GENERATOR_SET",
            None, None, np.zeros(3), False, True, "g", "c", "e", ("r", None),
            certificate_manifest_hash="manifest", backup_triggered=True,
            backup_reason="NO_GENERATOR_SET", energy=12.0, required_return_energy=7.0,
            energy_margin=4.0, charging=False, task_id="goal-1", goal_id="C",
            tasks_completed=2, recoverable_set_version="recoverable-set-v1",
            recoverability_action_rule_version="recoverability-action-rule-v1",
        )
        executed[0] = 0.9
        self.assertAlmostEqual(float(transition.executed_action[0]), 0.1)
        self.assertTrue(transition.backup_triggered)


if __name__ == "__main__":
    unittest.main()
