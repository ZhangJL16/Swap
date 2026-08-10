import unittest

import numpy as np

from envs.certified_uav import NavigationRewardConfig, PersistentNavigationEnv


class PersistentNavigationContractTests(unittest.TestCase):
    def make_env(self, **kwargs):
        return PersistentNavigationEnv(max_episode_steps=4, **kwargs)

    def test_A_inward_level_transition_is_positive(self):
        env = self.make_env()
        self.assertEqual(env.distance_potential(0.8), 2.0)
        self.assertEqual(env.distance_potential(0.65), 3.0)
        self.assertGreater(env.distance_potential_shaping(0.8, 0.65), 0.0)

    def test_B_outward_level_transition_is_negative(self):
        env = self.make_env()
        self.assertLess(env.distance_potential_shaping(0.65, 0.8), 0.0)

    def test_C_same_level_motion_has_only_discounted_potential_difference(self):
        env = self.make_env()
        expected = 0.25 * (0.99 * 3.0 - 3.0)
        self.assertAlmostEqual(env.distance_potential_shaping(0.65, 0.60), expected)
        env.reset(
            seed=1,
            options={
                "start_position": [1.0, 1.0, 1.0],
                "goal_position": [1.65, 1.0, 1.0],
                "start_velocity": [0.1, 0.0, 0.0],
            },
        )
        _, _, _, _, info = env.step(np.zeros(3))
        self.assertNotIn("goal_progress_reward", info["reward_components"])
        self.assertIn("distance_potential_shaping", info["reward_components"])

    def test_D_signed_velocity_toward_is_positive(self):
        env = self.make_env()
        value = env.signed_velocity_toward_goal(
            np.array([0.1, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([2.0, 1.0, 1.0]),
        )
        self.assertGreater(value, 0.0)

    def test_E_signed_velocity_away_is_negative(self):
        env = self.make_env()
        value = env.signed_velocity_toward_goal(
            np.array([-0.1, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([2.0, 1.0, 1.0]),
        )
        self.assertLess(value, 0.0)

    def test_F_orthogonal_velocity_is_zero(self):
        env = self.make_env()
        value = env.signed_velocity_toward_goal(
            np.array([0.0, 0.1, 0.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([2.0, 1.0, 1.0]),
        )
        self.assertAlmostEqual(value, 0.0, places=12)

    def test_G_approach_away_cycle_has_nonpositive_discounted_shaping_return(self):
        env = self.make_env(reward_config=NavigationRewardConfig(energy_cost_weight=0.0))
        inward_potential = env.distance_potential_shaping(0.8, 0.65)
        outward_potential = env.distance_potential_shaping(0.65, 0.8)
        toward = env.signed_velocity_toward_goal(
            np.array([0.1, 0.0, 0.0]),
            np.array([1.2, 1.0, 1.0]),
            np.array([2.0, 1.0, 1.0]),
        )
        away = env.signed_velocity_toward_goal(
            np.array([-0.1, 0.0, 0.0]),
            np.array([1.35, 1.0, 1.0]),
            np.array([2.0, 1.0, 1.0]),
        )
        inward_reward = inward_potential + 0.1 * toward - 0.01
        outward_reward = outward_potential + 0.1 * away - 0.01
        discounted_cycle_return = inward_reward + 0.99 * outward_reward
        self.assertLessEqual(discounted_cycle_return, 0.0)

    def test_H_goal_replacement_reward_uses_old_goal_only(self):
        env = self.make_env(reward_config=NavigationRewardConfig(energy_cost_weight=0.0))
        old_goal = np.array([1.0, 1.0, 1.0])
        env.reset(seed=2, options={"start_position": old_goal, "goal_position": old_goal})
        _, reward, terminated, truncated, info = env.step(np.zeros(3))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["task_completed_now"])
        np.testing.assert_array_equal(info["goal_before"], old_goal)
        np.testing.assert_array_equal(info["completed_goal"], old_goal)
        self.assertFalse(np.allclose(info["current_goal"], old_goal))
        self.assertEqual(info["distance_potential_before"], 4.0)
        self.assertEqual(info["distance_potential_after"], 4.0)
        self.assertAlmostEqual(info["reward_components"]["distance_potential_shaping"], -0.01)
        self.assertAlmostEqual(info["reward_components"]["task_completion_reward"], 10.0)
        self.assertAlmostEqual(reward, sum(info["reward_components"].values()))
        completion = info["goal_attempt_records"][0]
        np.testing.assert_array_equal(completion["goal_absolute_coordinates"], old_goal)

    def test_I_collision_penalizes_corrects_and_does_not_terminate(self):
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

    def test_J_velocity_saturation_clips_and_does_not_terminate(self):
        env = self.make_env()
        env.reset(
            seed=4,
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

    def test_K_only_max_steps_truncates_normal_episode(self):
        env = self.make_env()
        env.reset(seed=5, options={"start_position": [1.0, 1.0, 1.0], "goal_position": [3.0, 3.0, 1.0]})
        for index in range(4):
            _, _, terminated, truncated, _ = env.step(np.zeros(3))
            self.assertFalse(terminated)
            self.assertEqual(truncated, index == 3)

    def test_L_observation_contract_excludes_relative_and_certificate_fields(self):
        env = self.make_env()
        observation, _ = env.reset(
            seed=6,
            options={"start_position": [1.0, 1.5, 0.8], "goal_position": [3.0, 2.5, 1.2]},
        )
        self.assertEqual(
            env.observation_fields,
            (
                "absolute_position",
                "velocity",
                "absolute_goal_position",
                "absolute_station_position",
                "state_of_charge",
                "lidar_distances",
                "lidar_valid",
            ),
        )
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

    def test_M_action_contract_and_physical_scaling(self):
        env = self.make_env()
        self.assertEqual(env.action_space.shape, (3,))
        np.testing.assert_array_equal(env.action_space.low, -np.ones(3, dtype=np.float32))
        np.testing.assert_array_equal(env.action_space.high, np.ones(3, dtype=np.float32))
        physical = env.normalized_to_physical_action(np.array([1.0, -1.0, 0.5]))
        np.testing.assert_allclose(physical, np.array([0.18, -0.18, 0.04]))
        self.assertTrue(np.all(np.abs(physical) <= env.config.a_max + 1e-12))
        np.testing.assert_allclose(env.config.v_max, np.array([0.30, 0.30, 0.12]))
        self.assertEqual(env.config.dt, 0.2)

    def test_goal_attempt_records_completed_and_unfinished_goals(self):
        env = PersistentNavigationEnv(max_episode_steps=2)
        start = np.array([1.0, 1.0, 1.0])
        env.reset(seed=7, options={"start_position": start, "goal_position": start})
        _, _, _, truncated, first_info = env.step(np.zeros(3))
        self.assertFalse(truncated)
        completed = first_info["goal_attempt_records"][0]
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["steps_to_goal"], 1)
        required = {
            "goal_id",
            "goal_start_episode_step",
            "goal_absolute_coordinates",
            "goal_initial_distance",
            "initial_xy_distance",
            "initial_z_distance",
            "start_position",
            "final_distance",
            "collisions_during_goal",
            "boundary_contacts_during_goal",
            "velocity_saturations_during_goal",
            "mean_signed_velocity_toward_goal",
            "reward_component_totals",
        }
        self.assertTrue(required.issubset(completed))
        _, _, _, truncated, second_info = env.step(np.zeros(3))
        self.assertTrue(truncated)
        unfinished = second_info["goal_attempt_records"][-1]
        self.assertEqual(unfinished["status"], "unfinished_goal")
        self.assertFalse(unfinished["completed"])

    def test_boundary_lock_event_after_more_than_100_consecutive_contacts(self):
        env = PersistentNavigationEnv(max_episode_steps=120)
        radius = env.config.body_radius
        env.reset(
            seed=8,
            options={
                "start_position": [radius + 0.001, 1.0, 1.0],
                "goal_position": [3.0, 1.0, 1.0],
                "start_velocity": [-env.config.v_max[0], 0.0, 0.0],
            },
        )
        events = []
        for _ in range(101):
            _, _, terminated, truncated, info = env.step(np.array([-1.0, 0.0, 0.0]))
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            if info["boundary_lock_event"] is not None:
                events.append(info["boundary_lock_event"])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "BOUNDARY_LOCK_EVENT")
        self.assertEqual(events[0]["consecutive_boundary_contacts"], 101)
        self.assertEqual(info["maximum_consecutive_boundary_contacts"], 101)

    def test_reward_breakdown_has_only_formal_components(self):
        env = self.make_env()
        env.reset(seed=9, options={"start_position": [1.0, 1.0, 1.0], "goal_position": [3.0, 1.0, 1.0]})
        _, reward, _, _, info = env.step(np.zeros(3))
        self.assertEqual(
            tuple(info["reward_components"]),
            (
                "distance_potential_shaping",
                "signed_velocity_toward_goal_reward",
                "time_cost",
                "task_completion_reward",
                "collision_penalty",
                "energy_cost",
                "backup_intervention_event_cost",
            ),
        )
        self.assertAlmostEqual(reward, sum(info["reward_components"].values()))


if __name__ == "__main__":
    unittest.main()
