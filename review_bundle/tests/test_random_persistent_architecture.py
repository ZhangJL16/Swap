from __future__ import annotations

import unittest

import numpy as np

from envs.certified_uav import make_random_persistent_uav_env
from envs.certified_uav.persistent_task import PersistentMissionMode


class RandomPersistentArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.env = make_random_persistent_uav_env("random_persistent_open.json")

    def setUp(self) -> None:
        self.observation, self.info = self.env.reset(seed=7)

    def test_random_start_is_inside_certified_recoverable_domain(self) -> None:
        state = self.env.runtime._certificate_state()
        context = self.env.atlas.evaluate(state, self.env.plant.state.timestamp)
        self.assertTrue(self.env.atlas.contains_certificate_state(state))
        self.assertTrue(context.recovery.certified)
        self.assertTrue(self.env.atlas.last_recoverable_set_certificate.recoverable)

    def test_random_start_reproducible_by_seed(self) -> None:
        self.env.reset(seed=11)
        first = self.env.plant.state.copy()
        self.env.reset(seed=11)
        second = self.env.plant.state.copy()
        np.testing.assert_allclose(first.position, second.position)
        np.testing.assert_allclose(first.velocity, second.velocity)
        self.assertEqual(first.energy, second.energy)

    def test_random_goal_reproducible_by_seed(self) -> None:
        self.env.reset(seed=13)
        first = self.env.task_env.manager.current_task.goal_position.copy()
        self.env.reset(seed=13)
        second = self.env.task_env.manager.current_task.goal_position.copy()
        np.testing.assert_allclose(first, second)

    def test_random_goal_not_limited_to_fixed_goal_nodes(self) -> None:
        goals = []
        for seed in range(5):
            self.env.reset(seed=seed)
            goals.append(self.env.task_env.manager.current_task.goal_position.copy())
        self.assertFalse(hasattr(self.env.task_env.manager, "network"))
        self.assertGreater(len({tuple(np.round(goal, 6)) for goal in goals}), 1)
        self.assertTrue(any(not np.allclose(goal[:2] * 20.0, np.round(goal[:2] * 20.0)) for goal in goals))

    def _support_for_goal(self, goal: np.ndarray):
        self.env.task_env.manager.current_task.goal_position = goal.copy()
        state = self.env.runtime._certificate_state()
        context = self.env.atlas.evaluate(state, self.env.plant.state.timestamp)
        certificate = context.closure.zonotope_certificate
        self.assertIsNotNone(certificate)
        return context, np.asarray(certificate.zonotope.center), np.asarray(certificate.zonotope.generators)

    def test_goal_change_does_not_change_recoverability_certificate(self) -> None:
        state = self.env.plant.state.position.copy()
        first_goal = self.env.atlas.sample_goal(np.random.default_rng(31), state, 0.5)
        second_goal = self.env.atlas.sample_goal(np.random.default_rng(32), state, 0.5)
        first, _, _ = self._support_for_goal(first_goal)
        second, _, _ = self._support_for_goal(second_goal)
        self.assertEqual(first.required_energy, second.required_energy)
        self.assertEqual(first.recovery.certificate_hash, second.recovery.certificate_hash)
        self.assertEqual(self.env.atlas.atlas_hash, self.info["recovery_atlas_hash"])

    def test_goal_change_does_not_change_generator_support(self) -> None:
        state = self.env.plant.state.position.copy()
        first_goal = self.env.atlas.sample_goal(np.random.default_rng(41), state, 0.5)
        second_goal = self.env.atlas.sample_goal(np.random.default_rng(42), state, 0.5)
        _, first_c, first_g = self._support_for_goal(first_goal)
        _, second_c, second_g = self._support_for_goal(second_goal)
        np.testing.assert_allclose(first_c, second_c)
        np.testing.assert_allclose(first_g, second_g)

    def test_goal_change_can_change_actor_action(self) -> None:
        state = self.env.plant.state.position.copy()
        first_goal = self.env.atlas.sample_goal(np.random.default_rng(51), state, 0.5)
        second_goal = self.env.atlas.sample_goal(np.random.default_rng(52), state, 0.5)
        first_u = np.clip(first_goal - state, -1.0, 1.0)
        second_u = np.clip(second_goal - state, -1.0, 1.0)
        self.assertFalse(np.allclose(first_u, second_u))

    def test_main_method_has_no_task_edge_dependency(self) -> None:
        self.assertFalse(self.env.atlas.consumes_task_edges)
        self.assertFalse(self.env.task_env.task_edge_dependency)
        self.assertFalse(hasattr(self.env.task_env.manager, "active_route"))
        self.assertIsNone(self.env.network)

    def test_main_method_has_no_task_waypoint_dependency(self) -> None:
        self.assertFalse(self.env.atlas.consumes_task_waypoints)
        self.assertFalse(self.env.task_env.task_waypoint_dependency)
        self.assertFalse(hasattr(self.env.atlas, "task_reference"))
        self.assertFalse(hasattr(self.env.atlas, "task_waypoints"))

    def test_generator_successor_targets_recovery_atlas_not_task_reference(self) -> None:
        context = self.env.atlas.evaluate(self.env.runtime._certificate_state())
        self.assertIsNotNone(context.task_successor_cell_id)
        self.assertIn(context.task_successor_cell_id, self.env.atlas._cells_by_id)
        self.assertFalse(hasattr(self.env.atlas, "task_reference"))

    def test_random_goal_completion_samples_new_goal_without_resetting_plant(self) -> None:
        manager = self.env.task_env.manager
        previous = manager.current_task.goal_position.copy()
        plant_identity = id(self.env.plant)
        timestamp = self.env.plant.state.timestamp
        events = manager.advance(previous, step=17)
        self.assertTrue(events["task_completed"])
        self.assertEqual(manager.tasks_completed, 1)
        self.assertEqual(id(self.env.plant), plant_identity)
        self.assertEqual(self.env.plant.state.timestamp, timestamp)
        self.assertFalse(np.allclose(previous, manager.current_task.goal_position))

    def test_charging_preserves_same_pending_random_goal(self) -> None:
        task = self.env.task_env.manager.current_task
        task_id = task.task_id
        goal = task.goal_position.copy()
        self.env.task_env.enter_charging(voluntary=True)
        self.env.task_env.leave_station()
        self.assertEqual(self.env.task_env.manager.current_task.task_id, task_id)
        np.testing.assert_allclose(self.env.task_env.manager.current_task.goal_position, goal)

    def test_kappa_backup_preserves_pending_random_goal(self) -> None:
        task = self.env.task_env.manager.current_task
        task_id = task.task_id
        goal = task.goal_position.copy()
        self.env.task_env.begin_backup_recovery("TEST_BOUNDARY")
        self.assertEqual(self.env.task_env.mode, PersistentMissionMode.BACKUP_RECOVERY)
        self.assertEqual(self.env.task_env.manager.current_task.task_id, task_id)
        np.testing.assert_allclose(self.env.task_env.manager.current_task.goal_position, goal)

    def test_safe_departure_does_not_require_precomputed_task_path(self) -> None:
        first = self.env.atlas.required_departure_energy(self.env.task_env.manager.current_task)
        replacement = self.env.atlas.sample_goal(np.random.default_rng(61), self.env.plant.state.position, 0.5)
        self.env.task_env.manager.current_task.goal_position = replacement
        second = self.env.atlas.required_departure_energy(self.env.task_env.manager.current_task)
        self.assertEqual(first, second)
        self.assertFalse(hasattr(self.env.atlas, "network"))

    def test_no_generator_invokes_kappa_without_safety_violation(self) -> None:
        self.env.atlas.recovery_active = True
        context = self.env.atlas.evaluate(self.env.runtime._certificate_state())
        self.assertTrue(context.recovery.certified)
        self.assertIsNone(context.closure.zonotope_certificate)
        self.assertEqual(context.closure.status, "RECOVERY_TAKEOVER")

    def test_recovery_atlas_hash_does_not_depend_on_goal_seed(self) -> None:
        manifest = self.env.atlas.atlas_hash
        for seed in (0, 1, 2, 3, 4):
            _, info = self.env.reset(seed=seed)
            self.assertEqual(self.env.atlas.atlas_hash, manifest)
            self.assertEqual(info["recovery_atlas_hash"], manifest)


if __name__ == "__main__":
    unittest.main()
