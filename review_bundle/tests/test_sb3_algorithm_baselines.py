from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from envs.certified_uav import PersistentNavigationEnv
from scripts.sb3_navigation_harness import rollout_aligned_step_at_or_after
from scripts.train_sb3_navigation_baseline import build_model


class SB3AlgorithmBaselineTests(unittest.TestCase):
    def test_ppo_is_standard_sb3_and_rollout_checkpoint_is_aligned(self):
        args = SimpleNamespace(algorithm="ppo", seed=0, device="cpu", action_noise_sigma=0.1)
        environment = Monitor(PersistentNavigationEnv(max_episode_steps=8))
        model = build_model(args, environment)
        self.assertIsInstance(model, PPO)
        self.assertEqual(model.n_steps, 2048)
        self.assertEqual(model.batch_size, 64)
        self.assertEqual(model.n_epochs, 10)
        self.assertEqual(rollout_aligned_step_at_or_after(10_000, model.n_steps), 10_240)
        self.assertEqual(rollout_aligned_step_at_or_after(1_000_000, model.n_steps), 1_001_472)
        environment.close()

    def test_ddpg_is_standard_sb3_with_required_normal_action_noise(self):
        args = SimpleNamespace(algorithm="ddpg", seed=1, device="cpu", action_noise_sigma=0.1)
        environment = Monitor(PersistentNavigationEnv(max_episode_steps=8))
        model = build_model(args, environment)
        self.assertIsInstance(model, DDPG)
        self.assertIsInstance(model.action_noise, NormalActionNoise)
        np.testing.assert_allclose(model.action_noise._mu, np.zeros(3))
        np.testing.assert_allclose(model.action_noise._sigma, np.full(3, 0.1))
        environment.close()

    def test_ppo_and_ddpg_checkpoint_save_load_and_deterministic_predict(self):
        observation = np.zeros(77, dtype=np.float32)
        for algorithm, expected_type in (("ppo", PPO), ("ddpg", DDPG)):
            args = SimpleNamespace(algorithm=algorithm, seed=2, device="cpu", action_noise_sigma=0.1)
            environment = Monitor(PersistentNavigationEnv(max_episode_steps=8))
            model = build_model(args, environment)
            action, _ = model.predict(observation, deterministic=True)
            self.assertEqual(action.shape, (3,))
            self.assertTrue(np.all(np.isfinite(action)))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / algorithm
                model.save(path)
                loaded = expected_type.load(path, device="cpu")
                loaded_action, _ = loaded.predict(observation, deterministic=True)
                self.assertTrue(np.all(np.isfinite(loaded_action)))
            environment.close()


if __name__ == "__main__":
    unittest.main()
