import unittest

import numpy as np

from envs.certified_uav import NavigationRewardConfig, PersistentNavigationEnv


class PersistentNavigationContractTests(unittest.TestCase):
    def make_env(self, **kwargs):
        return PersistentNavigationEnv(max_episode_steps=4, **kwargs)

    def test_action_contract_and_physical_scaling(self):
        env = self.make_env()
        self.assertEqual(env.action_space.shape, (3,))
        np.testing.assert_array_equal(env.action_space.low, -np.ones(3, dtype=np.float32))
        np.testing.assert_array_equal(env.action_space.high, np.ones(3, dtype=np.float32))
        physical = env.normalized_to_physical_action(np.array([1.0, -1.0, 0.5]))
        self.assertTrue(np.all(np.abs(physical) <= env.config.a_max + 1e-12))
        np.testing.assert_allclose(physical, env.config.a_max * np.array([1.0, -1.0, 0.5]))

    def test_observation_uses_absolute_coordinates_without_certificate_or_relative_fields(self):
        env = self.make_env()
        observation, _ = env.reset(
            seed=1,
            options={"start_position": [1.0, 1.5, 0.8], "goal_position": [3.0, 2.5, 1.2]},
        )
        expected_fields = (
            "absolute_position",
            "velocity",
            "absolute_goal_position",
            "absolute_station_position",
            "state_of_charge",
            "lidar_distances",
            "lidar_valid",
        )
        self.assertEqual(env.observation_fields, expected_fields)
        forbidden = {
            "goal_delta",
            "station_delta",
            "distance_to_goal",
            "distance_to_station",
            "required_return_energy",
            "energy_margin",
            "recovery_corridor",
            "certificate_cell_identity",
            "generator_c",
            "generator_G",
            "kappa_internal_state",
        }
        self.assertTrue(forbidden.isdisjoint(env.observation_fields))
        np.testing.assert_allclose(
            observation[env.observation_layout["absolute_goal_position"]],
            np.array([3.0, 2.5, 1.2]) / env.config.world_size,
        )
        self.assertEqual(observation.shape, (77,))

    def test_goal_completion_assigns_new_goal_without_termination(self):
        env = self.make_env()
        start = np.array([1.0, 1.0, 1.0])
        env.reset(seed=2, options={"start_position": start, "goal_position": start})
        _, reward, terminated, truncated, info = env.step(np.zeros(3))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["task_completed_now"])
        self.assertEqual(info["tasks_completed"], 1)
        self.assertGreaterEqual(reward, env.reward_config.task_completion_reward - env.reward_config.time_cost - 0.01)
        self.assertGreater(np.linalg.norm(info["current_goal"] - start), env.minimum_goal_separation - 1e-12)

    def test_boundary_collision_penalizes_corrects_and_continues(self):
        env = self.make_env()
        radius = env.config.body_radius
        env.reset(
            seed=3,
            options={
                "start_position": [radius + 0.001, 1.0, 1.0],
                "goal_position": [3.0, 1.0, 1.0],
                "start_velocity": [-env.config.v_max[0], 0.0, 0.0],
            },
        )
        _, _, terminated, truncated, info = env.step(np.array([-1.0, 0.0, 0.0]))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["boundary_collision"])
        self.assertLess(info["reward_components"]["collision_penalty"], 0.0)
        self.assertTrue(env._is_legal_position(env.state.position))
        self.assertEqual(env.state.velocity[0], 0.0)

    def test_obstacle_collision_penalizes_corrects_and_continues(self):
        env = PersistentNavigationEnv(
            "random_persistent_obstacle.json",
            max_episode_steps=4,
        )
        obstacle = env.world.aabbs[0] if env.world.aabbs else None
        if obstacle is not None:
            start = np.array([obstacle.low[0] - env.config.body_radius - 0.01, np.mean(obstacle.low[1:2]), 1.0])
            goal = np.array([max(0.3, start[0] - 0.8), start[1], start[2]])
            action = np.array([1.0, 0.0, 0.0])
        else:
            cylinder = env.world.cylinders[0]
            start = np.array([cylinder.center_xy[0] - cylinder.radius - env.config.body_radius - 0.01, cylinder.center_xy[1], 1.0])
            goal = np.array([max(0.3, start[0] - 0.8), start[1], start[2]])
            action = np.array([1.0, 0.0, 0.0])
        env.reset(
            seed=4,
            options={"start_position": start, "goal_position": goal, "start_velocity": env.config.v_max * action},
        )
        _, _, terminated, truncated, info = env.step(action)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["obstacle_collision"])
        self.assertTrue(env._is_legal_position(env.state.position))
        np.testing.assert_array_equal(env.state.velocity, np.zeros(3))

    def test_velocity_saturation_does_not_terminate(self):
        env = self.make_env()
        env.reset(
            seed=5,
            options={
                "start_position": [2.0, 2.0, 1.0],
                "goal_position": [3.2, 2.0, 1.0],
                "start_velocity": env.config.v_max,
            },
        )
        _, _, terminated, truncated, info = env.step(np.ones(3))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["velocity_saturated"])
        self.assertTrue(np.all(np.abs(env.state.velocity) <= env.config.v_max + 1e-12))

    def test_only_max_steps_truncates_normal_episode(self):
        env = self.make_env()
        env.reset(seed=6, options={"start_position": [1.0, 1.0, 1.0], "goal_position": [3.0, 3.0, 1.0]})
        for index in range(4):
            _, _, terminated, truncated, _ = env.step(np.zeros(3))
            self.assertFalse(terminated)
            self.assertEqual(truncated, index == 3)

    def test_reward_breakdown_and_velocity_direction(self):
        reward_config = NavigationRewardConfig(energy_cost_weight=0.0)
        toward = self.make_env(reward_config=reward_config)
        toward.reset(
            seed=7,
            options={
                "start_position": [1.0, 1.0, 1.0],
                "goal_position": [3.0, 1.0, 1.0],
                "start_velocity": [0.1, 0.0, 0.0],
            },
        )
        _, reward, _, _, info = toward.step(np.zeros(3))
        components = info["reward_components"]
        self.assertAlmostEqual(reward, sum(components.values()))
        self.assertAlmostEqual(components["goal_progress_reward"], reward_config.progress_weight * info["goal_progress"])
        self.assertGreater(components["velocity_toward_goal_reward"], 0.0)
        self.assertEqual(components["time_cost"], -reward_config.time_cost)

        away = self.make_env(reward_config=reward_config)
        away.reset(
            seed=8,
            options={
                "start_position": [1.0, 1.0, 1.0],
                "goal_position": [3.0, 1.0, 1.0],
                "start_velocity": [-0.1, 0.0, 0.0],
            },
        )
        _, _, _, _, away_info = away.step(np.zeros(3))
        self.assertEqual(away_info["reward_components"]["velocity_toward_goal_reward"], 0.0)


if __name__ == "__main__":
    unittest.main()
