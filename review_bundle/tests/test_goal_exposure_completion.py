from __future__ import annotations

import unittest

import numpy as np

from cert_runtime.goal_exposure import task_completion_distance_invariant
from envs.certified_uav import make_random_persistent_uav_env


class GoalExposureCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.environment = make_random_persistent_uav_env("random_persistent_open.json", seed=0)

    def setUp(self) -> None:
        self.observation, self.reset_info = self.environment.reset(seed=17)

    def _place_goal_at_current_position(self) -> np.ndarray:
        goal = self.environment.plant.state.position.copy()
        self.environment.task_env.manager.current_task.goal_position = goal.copy()
        return goal

    def test_task_completes_when_entering_goal_radius_in_task_rl(self) -> None:
        self._place_goal_at_current_position()
        _, _, _, _, info = self.environment.task_env.step(np.zeros(3, dtype=np.float64))
        self.assertTrue(info["task_completed_now"])
        self.assertTrue(info["task_completion_distance_invariant"])

    def test_goal_radius_boundary_exact(self) -> None:
        manager = self.environment.task_env.manager
        goal = np.array((2.0, 2.0, 1.0), dtype=np.float64)
        manager.current_task.goal_position = goal.copy()
        events = manager.advance(goal + np.array((manager.goal_radius, 0.0, 0.0)), step=3)
        self.assertTrue(events["task_completed"])

    def test_goal_does_not_complete_in_backup_recovery(self) -> None:
        self._place_goal_at_current_position()
        self.environment.task_env.begin_backup_recovery("TEST")
        _, _, _, _, info = self.environment.task_env.step(np.zeros(3, dtype=np.float64))
        self.assertFalse(info["task_completed_now"])
        self.assertEqual(self.environment.task_env.manager.tasks_completed, 0)

    def test_goal_does_not_complete_in_charging_mode(self) -> None:
        self._place_goal_at_current_position()
        self.environment.task_env.enter_charging(voluntary=True)
        _, _, _, _, info = self.environment.task_env.step(np.zeros(3, dtype=np.float64))
        self.assertFalse(info["task_completed_now"])
        self.assertEqual(self.environment.task_env.manager.tasks_completed, 0)

    def test_completion_uses_pre_transition_pending_goal(self) -> None:
        goal = self._place_goal_at_current_position()
        _, _, _, _, info = self.environment.task_env.step(np.zeros(3, dtype=np.float64))
        np.testing.assert_allclose(info["goal_before"], goal)
        self.assertEqual(info["completed_task_id"], "random-goal-0")
        self.assertEqual(info["current_goal_id"], "random-goal-1")

    def test_completed_task_assigns_new_goal_after_reward_boundary(self) -> None:
        old_goal = self._place_goal_at_current_position()
        _, reward, _, _, info = self.environment.task_env.step(np.zeros(3, dtype=np.float64))
        self.assertTrue(info["task_completed_now"])
        self.assertGreater(info["reward_components"]["task_completion_reward"], 0.0)
        self.assertGreater(reward, 0.0)
        self.assertFalse(np.allclose(old_goal, info["current_goal"]))

    def test_minimum_task_distance_metric_uses_current_pending_goal(self) -> None:
        old_goal = self._place_goal_at_current_position()
        _, _, _, _, info = self.environment.task_env.step(np.zeros(3, dtype=np.float64))
        expected = np.linalg.norm(info["telemetry"].state_after.position - old_goal)
        self.assertAlmostEqual(info["distance_to_goal_after"], expected)

    def test_task_completion_distance_invariant(self) -> None:
        self.assertTrue(task_completion_distance_invariant("TASK_RL", 0.2, 0.2, True))
        self.assertFalse(task_completion_distance_invariant("TASK_RL", 0.1, 0.2, False))
        self.assertTrue(task_completion_distance_invariant("BACKUP_RECOVERY", 0.01, 0.2, False))

    def test_minimum_distance_metric_mode_semantics(self) -> None:
        self.assertTrue(task_completion_distance_invariant("KAPPA_BACKUP", 0.0064, 0.2, False))
        self.assertFalse(task_completion_distance_invariant("TASK_RL", 0.0064, 0.2, False))

if __name__ == "__main__":
    unittest.main()
