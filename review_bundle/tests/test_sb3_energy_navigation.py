import unittest

import numpy as np

from envs.certified_uav import PersistentEnergyNavigationEnv


class PersistentEnergyNavigationTests(unittest.TestCase):
    station = np.array([0.4, 0.5, 1.0])
    task_goal = np.array([3.0, 3.0, 1.0])

    def make_env(self, steps=8):
        return PersistentEnergyNavigationEnv(max_episode_steps=steps)

    def test_finite_battery_decrements_away_from_station(self):
        env = self.make_env()
        env.reset(
            seed=1,
            options={
                "start_position": [2.0, 2.0, 1.0],
                "goal_position": self.task_goal,
                "initial_energy_fraction": 1.0,
            },
        )
        before = env.state.energy
        _, _, terminated, truncated, info = env.step(np.zeros(3))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertLess(env.state.energy, before)
        self.assertGreater(info["flight_energy_used"], 0.0)
        self.assertEqual(info["gross_charge_received"], 0.0)

    def test_stationary_station_state_receives_point_four_energy(self):
        env = self.make_env()
        env.reset(
            seed=2,
            options={
                "start_position": self.station,
                "goal_position": self.task_goal,
                "initial_energy_fraction": 0.30,
            },
        )
        before = env.state.energy
        _, _, _, _, info = env.step(np.zeros(3))
        self.assertTrue(info["charging"])
        self.assertAlmostEqual(info["gross_charge_received"], 0.4)
        self.assertAlmostEqual(
            env.state.energy,
            before - info["flight_energy_used"] + info["gross_charge_received"],
        )

    def test_charging_is_clipped_at_battery_capacity(self):
        env = self.make_env()
        env.reset(
            seed=3,
            options={
                "start_position": self.station,
                "goal_position": self.task_goal,
                "initial_energy_fraction": 1.0,
            },
        )
        _, _, _, _, info = env.step(np.zeros(3))
        self.assertTrue(info["charging"])
        self.assertAlmostEqual(env.state.energy, 30.0)
        self.assertLess(info["gross_charge_received"], 0.4)

    def test_no_charge_outside_station(self):
        env = self.make_env()
        env.reset(
            seed=4,
            options={
                "start_position": [2.0, 2.0, 1.0],
                "goal_position": self.task_goal,
                "initial_energy_fraction": 0.30,
            },
        )
        _, _, _, _, info = env.step(np.zeros(3))
        self.assertFalse(info["inside_charging_region"])
        self.assertFalse(info["charging"])

    def test_no_charge_when_velocity_exceeds_gate(self):
        env = self.make_env()
        env.reset(
            seed=5,
            options={
                "start_position": self.station,
                "goal_position": self.task_goal,
                "start_velocity": [0.10, 0.0, 0.0],
                "initial_energy_fraction": 0.30,
            },
        )
        _, _, _, _, info = env.step(np.zeros(3))
        self.assertTrue(info["inside_charging_region"])
        self.assertFalse(info["charging"])
        self.assertEqual(info["gross_charge_received"], 0.0)

    def test_action_remains_three_dimensional_continuous_motion(self):
        env = self.make_env()
        self.assertEqual(env.action_space.shape, (3,))
        np.testing.assert_array_equal(env.action_space.low, -np.ones(3, dtype=np.float32))
        np.testing.assert_array_equal(env.action_space.high, np.ones(3, dtype=np.float32))

    def test_pending_goal_is_preserved_across_charge_and_departure(self):
        env = self.make_env()
        env.reset(
            seed=6,
            options={
                "start_position": self.station,
                "goal_position": self.task_goal,
                "initial_energy_fraction": 0.30,
            },
        )
        pending_goal = env.goal.copy()
        pending_goal_id = env._goal_attempt["goal_id"]
        _, _, _, _, first = env.step(np.zeros(3))
        self.assertTrue(first["charging"])
        np.testing.assert_array_equal(env.goal, pending_goal)
        self.assertEqual(first["current_goal_id"], pending_goal_id)

        env.step(np.array([1.0, 0.0, 0.0]))
        _, _, _, _, departure = env.step(np.array([1.0, 0.0, 0.0]))
        self.assertFalse(departure["charging"])
        np.testing.assert_array_equal(env.goal, pending_goal)
        self.assertEqual(departure["current_goal_id"], pending_goal_id)
        session = departure["charging_session_records"][0]
        self.assertEqual(session["interrupted_pending_goal_id"], pending_goal_id)
        self.assertEqual(session["resumed_pending_goal_id"], pending_goal_id)
        self.assertTrue(session["successful_resume"])

    def test_depletion_strands_without_teleport_or_termination(self):
        env = self.make_env()
        start = np.array([2.0, 2.0, 1.0])
        env.reset(
            seed=7,
            options={
                "start_position": start,
                "goal_position": self.task_goal,
                "initial_energy_fraction": 1e-6,
            },
        )
        _, _, terminated, truncated, info = env.step(np.zeros(3))
        stranded_position = env.state.position.copy()
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["energy_stranded"])
        self.assertEqual(info["stranding_event"]["event"], "ENERGY_STRANDED")
        self.assertGreater(np.linalg.norm(stranded_position - env.scenario.station_position), 1.0)
        _, _, terminated, _, info = env.step(np.ones(3))
        self.assertFalse(terminated)
        self.assertTrue(info["energy_stranded"])
        np.testing.assert_allclose(env.state.position, stranded_position)
        np.testing.assert_allclose(env.state.velocity, np.zeros(3))

    def test_zero_energy_at_station_can_recharge_without_motion(self):
        env = self.make_env()
        env.reset(
            seed=8,
            options={
                "start_position": self.station,
                "goal_position": self.task_goal,
                "initial_energy_fraction": 0.0,
            },
        )
        _, _, terminated, _, info = env.step(np.ones(3))
        self.assertFalse(terminated)
        self.assertTrue(info["charging"])
        self.assertFalse(info["energy_stranded"])
        self.assertAlmostEqual(env.state.energy, 0.4)
        np.testing.assert_allclose(env.state.position, self.station)

    def test_only_max_step_truncates_energy_episode(self):
        env = self.make_env(steps=3)
        env.reset(
            seed=9,
            options={
                "start_position": [2.0, 2.0, 1.0],
                "goal_position": self.task_goal,
                "initial_energy_fraction": 1e-6,
            },
        )
        for index in range(3):
            _, _, terminated, truncated, _ = env.step(np.zeros(3))
            self.assertFalse(terminated)
            self.assertEqual(truncated, index == 2)

    def test_full_soc_reset_and_random_soc_reproducibility(self):
        full = self.make_env()
        _, info = full.reset(seed=10, options={"initial_energy_fraction": 1.0})
        self.assertEqual(info["initial_soc"], 1.0)
        first = self.make_env()
        second = self.make_env()
        _, first_info = first.reset(seed=11)
        _, second_info = second.reset(seed=11)
        self.assertAlmostEqual(first_info["initial_soc"], second_info["initial_soc"])
        self.assertGreaterEqual(first_info["initial_soc"], 0.30)
        self.assertLessEqual(first_info["initial_soc"], 1.00)

    def test_observation_and_reward_contract_remain_clean(self):
        env = self.make_env()
        observation, _ = env.reset(seed=12)
        self.assertEqual(observation.shape, (77,))
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
        _, _, _, _, info = env.step(np.zeros(3))
        self.assertEqual(
            set(info["reward_components"]),
            {
                "distance_potential_shaping",
                "signed_velocity_toward_goal_reward",
                "time_cost",
                "task_completion_reward",
                "collision_penalty",
                "energy_cost",
                "backup_intervention_event_cost",
            },
        )


if __name__ == "__main__":
    unittest.main()
