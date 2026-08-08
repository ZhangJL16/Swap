from __future__ import annotations

from itertools import product
import unittest

import numpy as np

from cert_runtime.task_authority import (
    BestInGeneratorGoalOracle,
    CenterOnlyGoalController,
    MaxOpposeCenterOracle,
    RandomInGeneratorGoalController,
    action_from_eta,
    support_authority_metrics,
    support_interval,
)
from envs.certified_uav import make_random_persistent_uav_env


class TaskAuthorityPureTests(unittest.TestCase):
    def setUp(self):
        self.center = np.array((0.03, -0.02, 0.01), dtype=np.float64)
        self.generators = np.diag((0.009, 0.008, 0.004))

    def test_goal_direction_support_function_matches_corner_enumeration(self):
        direction = np.array((0.4, -0.7, 0.2), dtype=np.float64)
        direction /= np.linalg.norm(direction)
        minimum, maximum, _ = support_interval(self.center, self.generators, direction)
        projections = [
            float(direction @ action_from_eta(self.center, self.generators, np.asarray(eta)))
            for eta in product((-1.0, 1.0), repeat=3)
        ]
        self.assertAlmostEqual(minimum, min(projections))
        self.assertAlmostEqual(maximum, max(projections))

    def test_center_only_equals_eta_zero(self):
        eta = CenterOnlyGoalController().select_eta(None, np.ones(3), self.center, self.generators, 0.2)
        self.assertTrue(np.array_equal(eta, np.zeros(3)))
        self.assertTrue(np.allclose(action_from_eta(self.center, self.generators, eta), self.center))

    def test_random_in_generator_inside_support(self):
        controller = RandomInGeneratorGoalController(3)
        for _ in range(100):
            eta = controller.select_eta(None, np.ones(3), self.center, self.generators, 0.2)
            self.assertTrue(np.all(np.abs(eta) < 1.0))
            self.assertTrue(np.allclose(action_from_eta(self.center, self.generators, eta), self.center + self.generators @ eta))

    def test_anti_center_oracle_minimizes_center_projection(self):
        eta = MaxOpposeCenterOracle().select_eta(None, np.ones(3), self.center, self.generators, 0.2)
        selected = float(action_from_eta(self.center, self.generators, eta) @ self.center)
        corners = [
            float(action_from_eta(self.center, self.generators, np.asarray(candidate)) @ self.center)
            for candidate in product((-1.0, 1.0), repeat=3)
        ]
        self.assertAlmostEqual(selected, min(corners), places=7)

    def test_bidirectional_metrics_match_interval(self):
        metrics = support_authority_metrics(np.zeros(3), self.generators, np.array((1.0, 1.0, 0.0)))
        self.assertTrue(metrics.bidirectional_goal_authority)
        self.assertTrue(metrics.bidirectional_x)
        self.assertTrue(metrics.bidirectional_y)


class TaskAuthorityEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environment = make_random_persistent_uav_env("random_persistent_open.json", seed=0)

    def setUp(self):
        self.environment.reset(seed=0)

    def _context_for_goal(self, goal: np.ndarray):
        self.environment.task_env.manager.current_task.goal_position = goal.copy()
        self.environment.atlas.reset()
        return self.environment._refresh_context()

    def test_best_in_generator_action_inside_certified_support(self):
        context = self.environment._refresh_context()
        state = self.environment.plant.state.copy()
        goal = self.environment.task_env.manager.current_task.goal_position.copy()
        eta = BestInGeneratorGoalOracle().select_eta(
            state, goal, context["c"], context["G"], self.environment.plant.config.dt
        )
        self.assertTrue(np.all(np.abs(eta) <= 1.0))
        action = action_from_eta(context["c"], context["G"], eta)
        lower = np.asarray(context["c"]) - np.sum(np.abs(np.asarray(context["G"])), axis=1)
        upper = np.asarray(context["c"]) + np.sum(np.abs(np.asarray(context["G"])), axis=1)
        self.assertTrue(np.all(action >= lower - 1e-12))
        self.assertTrue(np.all(action <= upper + 1e-12))

    def test_task_authority_metrics_do_not_change_certificate(self):
        first = self._context_for_goal(np.array((0.8, 3.0, 1.0)))
        second = self._context_for_goal(np.array((3.0, 0.8, 1.0)))
        self.assertEqual(first["recovery_hash"], second["recovery_hash"])
        self.assertTrue(np.allclose(first["c"], second["c"]))
        self.assertTrue(np.allclose(first["G"], second["G"]))
        self.assertEqual(first["continuation_target_cell_id"], second["continuation_target_cell_id"])

    def test_task_authority_audit_is_goal_conditioned_only_in_controller(self):
        position = self.environment.plant.state.position.copy()
        first_goal = position + np.array((1.0, 0.0, 0.0))
        second_goal = position - np.array((1.0, 0.0, 0.0))
        first = self._context_for_goal(first_goal)
        second = self._context_for_goal(second_goal)
        state = self.environment.plant.state.copy()
        oracle = BestInGeneratorGoalOracle()
        first_eta = oracle.select_eta(state, first_goal, first["c"], first["G"], self.environment.plant.config.dt)
        second_eta = oracle.select_eta(state, second_goal, second["c"], second["G"], self.environment.plant.config.dt)
        self.assertFalse(np.array_equal(first_eta, second_eta))
        self.assertTrue(np.allclose(first["c"], second["c"]))
        self.assertTrue(np.allclose(first["G"], second["G"]))

    def test_multi_successor_viability_is_goal_independent(self):
        before = dict(self.environment.atlas._rl_successor_options)
        self._context_for_goal(np.array((0.8, 3.0, 1.0)))
        middle = dict(self.environment.atlas._rl_successor_options)
        self._context_for_goal(np.array((3.0, 0.8, 1.0)))
        self.assertEqual(before, middle)
        self.assertEqual(before, self.environment.atlas._rl_successor_options)
        self.assertTrue(any(len(options) > 1 for options in before.values()))

    def test_zero_or_stabilizing_center_contains_no_goal_information(self):
        first = self._context_for_goal(np.array((0.8, 3.0, 1.0)))
        second = self._context_for_goal(np.array((3.0, 0.8, 1.0)))
        self.assertTrue(np.allclose(first["c"], second["c"]))
        state = self.environment.runtime._certificate_state()
        cell = self.environment.atlas._locate_recoverable_cell(state)
        expected = (
            -self.environment.atlas.position_gain * (self.environment.plant.state.position - np.asarray(cell.reference_position))
            -self.environment.atlas.velocity_gain * self.environment.plant.state.velocity
        )
        self.assertTrue(np.allclose(first["c"], expected))

    def test_reference_action_not_used_for_normal_motion(self):
        context = self.environment._refresh_context()
        cell = self.environment.atlas._locate_recoverable_cell(self.environment.runtime._certificate_state())
        self.assertIsNotNone(cell)
        expected = (
            -self.environment.atlas.position_gain * (self.environment.plant.state.position - np.asarray(cell.reference_position))
            -self.environment.atlas.velocity_gain * self.environment.plant.state.velocity
        )
        self.assertTrue(np.allclose(context["c"], expected))
        self.assertFalse(np.array_equal(np.asarray(cell.reference_action), context["c"]))

    def test_complete_generator_still_inside_A_safe_and_A_cont(self):
        context = self.environment._refresh_context()
        self.assertTrue(context["recoverability_action_verified"])
        self.assertTrue(context["continuation_action_verified"])
        self.assertTrue(context["policy_authority_pass"])


if __name__ == "__main__":
    unittest.main()
