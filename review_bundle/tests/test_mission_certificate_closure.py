from __future__ import annotations

import unittest

import numpy as np

from envs.certified_uav import MissionTerminationReason, make_certified_uav_env


class MissionCertificateClosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.environment = make_certified_uav_env(
            "mission_open.json",
            timing_mode="functional",
        )

    def setUp(self) -> None:
        self.environment.reset(seed=0)

    def test_open_mission_has_corridor_wide_hash_linked_certificate(self) -> None:
        provider = self.environment.mission_provider
        self.assertTrue(provider.gate_pass)
        self.assertTrue(provider.manifest.hash_chain_valid)
        self.assertTrue(provider.manifest.cells)
        self.assertTrue(all(cell.complete_successor_containment for cell in provider.manifest.cells))
        self.assertTrue(all(cell.e3_residual >= 0.0 for cell in provider.manifest.cells))

    def test_task_oriented_center_is_not_recovery_action(self) -> None:
        context = self.environment.action_context()
        self.assertTrue(context["generator_available"])
        self.assertGreater(float(context["c"][1]), 0.0)
        self.assertLess(float(context["kappa"][0]), 0.0)
        self.assertFalse(np.allclose(context["c"], context["kappa"]))

    def test_deterministic_generator_mission_completes_and_returns(self) -> None:
        for _ in range(self.environment.config.episode_limit):
            _, _, terminated, truncated, info = self.environment.step(np.zeros(3))
            if terminated or truncated:
                break
        self.assertTrue(info["task_completed"])
        self.assertTrue(info["terminal_return_success"])
        self.assertEqual(info["mission_termination_reason"], MissionTerminationReason.TASK_AND_RETURN_SUCCESS.value)
        self.assertNotEqual(self.environment.plant.state.position.tolist(), self.environment.scenario.initial_state.position.tolist())

    def test_generator_varies_along_outbound_trajectory(self) -> None:
        centers, generators, volumes = set(), set(), set()
        for _ in range(16):
            context = self.environment.action_context()
            if context["G"] is not None:
                centers.add(tuple(np.round(context["c"], 8)))
                generators.add(tuple(np.round(context["G"].reshape(-1), 8)))
                volumes.add(round(8.0 * abs(float(np.linalg.det(context["G"]))), 12))
            self.environment.step(np.zeros(3))
        self.assertGreater(len(centers), 1)
        self.assertGreater(len(generators), 1)
        self.assertGreater(len(volumes), 1)

    def test_shield_rejection_uses_certified_kappa_not_generator_membership(self) -> None:
        provider = self.environment.mission_provider
        expected_hash = self.environment.action_context()["recovery_hash"]
        _, _, _, _, info = self.environment.step_nominal_action(-np.asarray(self.environment.config.a_max))
        self.assertFalse(info["accepted"])
        self.assertEqual(self.environment.last_preparation.recovery.certificate_hash, expected_hash)
        self.assertTrue(provider.recovery_active)
        self.assertFalse(info["telemetry"].collision)

    def test_narrow_exclusion_removes_generator_but_not_kappa_certificate(self) -> None:
        environment = make_certified_uav_env("mission_narrow.json")
        environment.reset(seed=0)
        root = next(
            cell
            for cell in environment.mission_provider.root_cells
            if 2.70 <= cell.reference_position[0] <= 2.90 and 1.50 <= cell.reference_position[1] <= 2.30
        )
        environment.plant.state.position[:] = root.reference_position
        environment.plant.state.velocity[:] = root.reference_velocity
        preparation = environment.prepare_certificate_cycle()
        self.assertTrue(preparation.recovery.certified)
        self.assertEqual(preparation.closure_result.status, "NO_GENERATOR_SET")
        self.assertIsNone(preparation.closure_result.zonotope_certificate)


if __name__ == "__main__":
    unittest.main()
