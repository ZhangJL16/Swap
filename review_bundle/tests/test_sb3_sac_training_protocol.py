import json
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from scripts.train_sb3_persistent_sac import _jsonable, _policy_sampling_seed, evaluate_model


class RandomPolicy:
    def predict(self, observation, deterministic):
        del observation
        if deterministic:
            return np.zeros(3, dtype=np.float32), None
        return (2.0 * torch.rand(3) - 1.0).numpy().astype(np.float32), None


class SB3SACTrainingProtocolTests(unittest.TestCase):
    def make_args(self):
        return SimpleNamespace(
            scenario="random_persistent_open.json",
            seed=2,
            max_episode_steps=8,
            navigation_energy_capacity=1000.0,
            goal_radius=0.20,
            minimum_goal_separation=0.60,
            sampling_margin=0.20,
            distance_potential_scale=0.25,
            velocity_reward_weight=0.1,
            time_cost=0.01,
            completion_reward=10.0,
            collision_penalty=1.2,
            energy_cost_weight=0.01,
            backup_intervention_cost=0.1,
            gamma=0.99,
            heldout_seeds=[100, 101],
            evaluation_steps=8,
            stochastic_policy_seed_base=73_000_000,
        )

    def test_stochastic_policy_seed_is_explicit_and_stream_specific(self):
        args = self.make_args()
        self.assertEqual(_policy_sampling_seed(args, 10_000, 100), 93_100_100)
        self.assertNotEqual(
            _policy_sampling_seed(args, 10_000, 100),
            _policy_sampling_seed(args, 10_000, 101),
        )

    def test_stochastic_evaluation_is_reproducible_and_restores_rng(self):
        args = self.make_args()
        model = RandomPolicy()
        torch.manual_seed(17)
        np.random.seed(19)
        torch_state_before = torch.random.get_rng_state().clone()
        numpy_state_before = np.random.get_state()

        first = evaluate_model(model, args, 10_000, deterministic=False)
        torch.testing.assert_close(torch.random.get_rng_state(), torch_state_before)
        numpy_state_after = np.random.get_state()
        self.assertEqual(numpy_state_after[0], numpy_state_before[0])
        np.testing.assert_array_equal(numpy_state_after[1], numpy_state_before[1])
        self.assertEqual(numpy_state_after[2:], numpy_state_before[2:])

        second = evaluate_model(model, args, 10_000, deterministic=False)
        self.assertEqual(
            json.dumps(_jsonable(first), sort_keys=True),
            json.dumps(_jsonable(second), sort_keys=True),
        )
        self.assertEqual(first["evaluation_mode"], "stochastic")
        self.assertTrue(all(row["policy_sampling_seed"] is not None for row in first["seed_results"]))

    def test_deterministic_and_stochastic_results_are_separate(self):
        args = self.make_args()
        model = RandomPolicy()
        deterministic = evaluate_model(model, args, 10_000, deterministic=True)
        stochastic = evaluate_model(model, args, 10_000, deterministic=False)
        self.assertEqual(deterministic["evaluation_mode"], "deterministic")
        self.assertEqual(stochastic["evaluation_mode"], "stochastic")
        self.assertTrue(all(row["policy_sampling_seed"] is None for row in deterministic["seed_results"]))
        self.assertTrue(all(row["policy_sampling_seed"] is not None for row in stochastic["seed_results"]))


if __name__ == "__main__":
    unittest.main()
