from __future__ import annotations

import unittest

import numpy as np

from cert_runtime.state import CertificateState
from envs.certified_uav import make_random_persistent_uav_env
from envs.certified_uav.persistent_task import PersistentMissionMode
from envs.certified_uav.state import UAVPhysicalState
from scripts.validate_random_persistent_authority_lifecycle import validate_scenario


class PersistentAuthorityDomainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environment = make_random_persistent_uav_env("random_persistent_open.json", seed=0)
        cls.environment.reset(seed=0)
        cls.atlas = cls.environment.atlas

    def setUp(self):
        self.environment.reset(seed=0)

    def _state_for_cell(self, cell, energy: float = 30.0) -> CertificateState:
        base = self.environment.runtime._certificate_state()
        return CertificateState(
            cell.reference_position,
            cell.reference_velocity,
            energy,
            base.charging_position,
            base.local_geometry,
            base.return_corridor,
            dict(base.explicit_task_state),
            base.position_error_radius,
            base.velocity_error_radius,
            base.energy_error_radius,
            dict(base.bound_versions),
        )

    def test_R_RL_is_subset_of_R(self):
        recoverable_ids = {cell.cell_id for cell in self.atlas.manifest.cells}
        self.assertTrue(self.atlas._rl_authority_cell_ids)
        self.assertTrue(self.atlas._rl_authority_cell_ids.issubset(recoverable_ids))

    def test_kappa_only_state_remains_recoverable(self):
        root_ids = self.atlas._rl_authority_cell_ids
        cell = next(cell for cell in self.atlas.manifest.cells if cell.cell_id not in root_ids and cell.level > 0)
        state = self._state_for_cell(cell)
        self.assertTrue(self.atlas.contains_certificate_state(state))
        self.assertNotIn(cell.cell_id, root_ids)

    def test_R_RL_cells_have_nondegenerate_generator(self):
        for cell in (self.atlas.rl_authority_cells[0], self.atlas.rl_authority_cells[len(self.atlas.rl_authority_cells) // 2]):
            state = self._state_for_cell(cell)
            self.atlas.reset()
            context = self.atlas.evaluate(state)
            certificate = context.closure.zonotope_certificate
            self.assertIsNotNone(certificate)
            self.assertGreaterEqual(certificate.zonotope.sigma_min_lower_bound, self.environment.runtime.config.minimum_generator_sigma - 1e-12)

    def test_normal_generator_successor_remains_in_R_RL_or_charge(self):
        state = self.environment.runtime._certificate_state()
        self.atlas.reset()
        context = self.atlas.evaluate(state)
        self.assertTrue(self.atlas.last_continuation_verified)
        self.assertIn(context.task_successor_cell_id, self.atlas._rl_authority_cell_ids)

    def test_goal_change_does_not_change_R_RL(self):
        before = (self.atlas._rl_authority_cell_ids, dict(self.atlas._rl_successor_ids), self.atlas.atlas_hash)
        self.environment.task_env.manager.current_task.goal_position = np.array([0.9, 3.0, 1.0])
        self.atlas.reset()
        self.atlas.evaluate(self.environment.runtime._certificate_state())
        after = (self.atlas._rl_authority_cell_ids, dict(self.atlas._rl_successor_ids), self.atlas.atlas_hash)
        self.assertEqual(before, after)

    def test_goal_change_does_not_change_continuation_support(self):
        state = self.environment.runtime._certificate_state()
        supports = []
        for goal in (np.array([0.9, 3.0, 1.0]), np.array([3.0, 1.0, 1.0])):
            self.environment.task_env.manager.current_task.goal_position = goal
            self.atlas.reset()
            context = self.atlas.evaluate(state)
            zonotope = context.closure.zonotope_certificate.zonotope
            supports.append((np.asarray(zonotope.center), np.asarray(zonotope.generators), context.task_successor_cell_id))
        self.assertTrue(np.allclose(supports[0][0], supports[1][0]))
        self.assertTrue(np.allclose(supports[0][1], supports[1][1]))
        self.assertEqual(supports[0][2], supports[1][2])

    def test_zero_step_station_recovery_certificate_valid(self):
        certificate = self.atlas.terminal_recovery_certificate
        self.assertTrue(certificate.valid)
        self.assertEqual(certificate.level, 0)
        self.assertIsNone(certificate.successor)
        self.assertEqual(certificate.recovery_energy_upper, 0.0)

    def test_station_terminal_energy_is_finite(self):
        self.environment.plant.state = UAVPhysicalState(
            self.environment.plant.scenario.station_position.copy(), np.zeros(3), 2.0, 0.0
        )
        self.environment.task_env.mode = PersistentMissionMode.CHARGING_RL
        self.environment.task_env.phase = self.environment.task_env.mode
        self.atlas.reset()
        context = self.environment._refresh_context()
        self.assertTrue(context["certificate_valid"])
        self.assertTrue(np.isfinite(context["recovery_energy_required"]))
        self.assertTrue(np.isfinite(context["energy_margin"]))

    def test_backup_arrival_does_not_invalidate_kappa_evidence(self):
        self.environment.plant.state = UAVPhysicalState(
            self.environment.plant.scenario.station_position.copy(), np.zeros(3), 2.0, 0.0
        )
        self.environment.task_env.mode = PersistentMissionMode.CHARGING_RL
        self.environment.task_env.phase = self.environment.task_env.mode
        self.atlas.recovery_active = True
        self.atlas.active_cell_id = None
        context = self.environment._refresh_context()
        self.assertTrue(context["certificate_valid"])
        self.assertEqual(context["recovery_hash"], self.atlas.terminal_recovery_certificate.certificate_hash)
        self.assertNotEqual(context["execution_authority_reason"], "KAPPA_CERTIFICATE_INVALID")

    def test_departure_successor_returns_to_R_RL(self):
        self.environment.plant.state = UAVPhysicalState(
            self.environment.plant.scenario.station_position.copy(), np.zeros(3), 30.0, 0.0
        )
        self.environment.task_env.mode = PersistentMissionMode.CHARGING_RL
        self.environment.task_env.phase = self.environment.task_env.mode
        self.atlas.reset()
        for _ in range(80):
            _, _, terminated, truncated, _ = self.environment.step(np.zeros(3))
            self.assertFalse(terminated or truncated)
            if self.environment.task_env.mode == PersistentMissionMode.TASK_RL:
                break
        self.assertEqual(self.environment.task_env.mode, PersistentMissionMode.TASK_RL)
        context = self.environment._refresh_context()
        self.assertTrue(context["rl_authority_set_member"])

    def test_no_generator_is_not_safety_failure(self):
        state = self.environment.runtime._certificate_state()
        context = self.atlas.evaluate(state)
        self.assertTrue(context.recovery.certified)
        self.assertTrue(self.atlas.last_recoverable_set_certificate.recoverable)

    def test_normal_RL_does_not_fall_into_kappa_only_cell_via_accepted_generator(self):
        for _ in range(100):
            _, _, terminated, truncated, info = self.environment.step(np.zeros(3))
            self.assertFalse(terminated or truncated)
            self.assertTrue(info["accepted"])
            context = self.environment._refresh_context()
            self.assertTrue(context["rl_authority_set_member"])
            self.assertTrue(context["continuation_action_verified"])


class PersistentAuthorityLifecycleEndToEndTests(unittest.TestCase):
    def test_random_persistent_full_authority_lifecycle(self):
        result = validate_scenario("random_persistent_open", (0,), 5)
        self.assertEqual(result["AUTHORITY_LIFECYCLE_GATE"], "PASS", result["failures"])
        trace = result["lifecycle_traces"][0]
        self.assertFalse(trace["invalid_kappa"])
        self.assertFalse(trace["fail_closed"])
        self.assertTrue(trace["pending_goal_preserved"])
        self.assertTrue(trace["post_departure_in_R_RL"])


if __name__ == "__main__":
    unittest.main()
