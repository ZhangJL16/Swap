from __future__ import annotations

import unittest

import numpy as np
import torch
from torch.distributions import Normal

from cert_runtime.generator_sac import GeneratorSACConfig, PersistentGeneratorSAC
from cert_runtime.optimization_diagnostics import entropy_decomposition
from cert_runtime.persistent_authority import ExecutionAuthority
from envs.certified_uav import make_random_persistent_uav_env
from envs.certified_uav.persistent_task import PersistentMissionMode
from envs.certified_uav.persistent_wrapper import backup_intervention_started


class BackupRewardEventTests(unittest.TestCase):
    def test_backup_penalty_is_event_not_occupancy(self):
        self.assertTrue(backup_intervention_started(PersistentMissionMode.TASK_RL, ExecutionAuthority.KAPPA_BACKUP))
        self.assertFalse(backup_intervention_started(PersistentMissionMode.BACKUP_RECOVERY, ExecutionAuthority.KAPPA_BACKUP))

    def test_backup_event_reward_does_not_repeat_during_recovery(self):
        sequence = (
            (PersistentMissionMode.TASK_RL, ExecutionAuthority.KAPPA_BACKUP),
            (PersistentMissionMode.BACKUP_RECOVERY, ExecutionAuthority.KAPPA_BACKUP),
            (PersistentMissionMode.BACKUP_RECOVERY, ExecutionAuthority.KAPPA_BACKUP),
        )
        self.assertEqual(sum(backup_intervention_started(*item) for item in sequence), 1)

    def test_second_backup_event_is_penalized_again(self):
        sequence = (
            (PersistentMissionMode.TASK_RL, ExecutionAuthority.KAPPA_BACKUP),
            (PersistentMissionMode.BACKUP_RECOVERY, ExecutionAuthority.KAPPA_BACKUP),
            (PersistentMissionMode.CHARGING_RL, ExecutionAuthority.CHARGER_CONSTRAINED),
            (PersistentMissionMode.TASK_RL, ExecutionAuthority.RL_GENERATOR),
            (PersistentMissionMode.TASK_RL, ExecutionAuthority.KAPPA_BACKUP),
        )
        self.assertEqual(sum(backup_intervention_started(*item) for item in sequence), 2)

    def test_backup_recovery_still_pays_elapsed_and_energy_cost(self):
        environment = make_random_persistent_uav_env("random_persistent_open.json", seed=0)
        environment.reset(seed=0)
        environment.task_env.mode = PersistentMissionMode.BACKUP_RECOVERY
        environment.task_env.phase = environment.task_env.mode
        _, reward, terminated, truncated, info = environment.task_env.step(np.zeros(3))
        self.assertFalse(terminated or truncated)
        components = info["reward_components"]
        self.assertLess(components["elapsed_time_cost"], 0.0)
        self.assertLessEqual(components["energy_cost"], 0.0)
        self.assertEqual(components["backup_intervention_event_cost"], 0.0)
        self.assertAlmostEqual(sum(components.values()), reward, places=7)

    def test_reward_decomposition_after_event_semantics(self):
        environment = make_random_persistent_uav_env("random_persistent_open.json", seed=0)
        environment.reset(seed=0)
        _, reward, terminated, truncated, info = environment.step(np.zeros(3))
        self.assertFalse(terminated or truncated)
        self.assertEqual(
            set(info["reward_components"]),
            {
                "goal_progress_reward",
                "task_completion_reward",
                "elapsed_time_cost",
                "energy_cost",
                "backup_intervention_event_cost",
                "charging_dwell_cost",
            },
        )
        self.assertAlmostEqual(sum(info["reward_components"].values()), reward, places=7)


class TemperatureCoordinateTests(unittest.TestCase):
    def setUp(self):
        self.distribution = Normal(
            torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float32),
            torch.tensor([[0.8, 0.9, 1.1]], dtype=torch.float32),
        )
        self.u = torch.tensor([[0.4, -0.2, 0.1]], dtype=torch.float32)
        self.generator = torch.diag_embed(torch.tensor([[0.04, 0.03, 0.02]], dtype=torch.float32))

    def _agent(self, coordinate: str) -> PersistentGeneratorSAC:
        return PersistentGeneratorSAC(
            6,
            GeneratorSACConfig(hidden_dim=8, temperature_coordinate=coordinate),
            seed=0,
        )

    def test_temperature_coordinate_config(self):
        self.assertEqual(GeneratorSACConfig().temperature_coordinate, "physical")
        self.assertEqual(GeneratorSACConfig(temperature_coordinate="normalized").temperature_coordinate, "normalized")
        with self.assertRaises(ValueError):
            GeneratorSACConfig(temperature_coordinate="invalid")

    def test_physical_temperature_uses_physical_log_prob(self):
        terms = entropy_decomposition(self.distribution, self.u, self.generator)
        self.assertTrue(torch.equal(self._agent("physical")._temperature_log_probability(terms), terms.physical_log_prob))

    def test_normalized_temperature_uses_normalized_log_prob(self):
        terms = entropy_decomposition(self.distribution, self.u, self.generator)
        self.assertTrue(torch.equal(self._agent("normalized")._temperature_log_probability(terms), terms.normalized_log_prob))

    def test_actor_loss_density_semantics_are_physical_in_both_modes(self):
        terms = entropy_decomposition(self.distribution, self.u, self.generator)
        for coordinate in ("physical", "normalized"):
            self.assertTrue(torch.equal(self._agent(coordinate)._actor_log_probability(terms), terms.physical_log_prob))

    def test_bellman_density_is_physical_in_both_modes(self):
        observations = torch.zeros((1, 6), dtype=torch.float32)
        centers = torch.zeros((1, 3), dtype=torch.float32)
        for coordinate in ("physical", "normalized"):
            agent = self._agent(coordinate)
            torch.manual_seed(9)
            _, sampled_log_prob, sampled_u = agent._sample_generator_actions(observations, centers, self.generator)
            distribution = agent.actor.distribution(observations)
            expected = entropy_decomposition(distribution, sampled_u, self.generator).physical_log_prob
            self.assertTrue(torch.allclose(sampled_log_prob, expected))

    def test_normalized_alpha_residual_affine_scale_invariant(self):
        base = entropy_decomposition(self.distribution, self.u, self.generator)
        for factor in (0.5, 2.0):
            scaled = entropy_decomposition(self.distribution, self.u, factor * self.generator)
            self.assertTrue(torch.allclose(base.normalized_log_prob - 3.0, scaled.normalized_log_prob - 3.0))

    def test_physical_alpha_residual_has_expected_logdet_shift(self):
        base = entropy_decomposition(self.distribution, self.u, self.generator)
        doubled = entropy_decomposition(self.distribution, self.u, 2.0 * self.generator)
        expected = torch.tensor([3.0 * np.log(2.0)], dtype=torch.float32)
        self.assertTrue(torch.allclose((base.physical_log_prob - 3.0) - (doubled.physical_log_prob - 3.0), expected, atol=1e-6))

    def test_temperature_mode_does_not_change_executed_action_mapping(self):
        center = torch.tensor([[0.01, -0.02, 0.0]], dtype=torch.float32)
        for coordinate in ("physical", "normalized"):
            action = self._agent(coordinate)._mapped_action(center, self.generator, self.u)
            expected = center + torch.bmm(self.generator, torch.tanh(self.u).unsqueeze(-1)).squeeze(-1)
            self.assertTrue(torch.allclose(action, expected))

    def test_temperature_mode_does_not_change_safe_support_or_kappa(self):
        environment = make_random_persistent_uav_env("random_persistent_open.json", seed=3)
        _, info = environment.reset(seed=3)
        context = info["action_context"]
        support = (np.asarray(context["c"]).copy(), np.asarray(context["G"]).copy(), np.asarray(context["kappa"]).copy())
        for coordinate in ("physical", "normalized"):
            self._agent(coordinate)
            refreshed = environment._refresh_context()
            np.testing.assert_allclose(refreshed["c"], support[0])
            np.testing.assert_allclose(refreshed["G"], support[1])
            np.testing.assert_allclose(refreshed["kappa"], support[2])


if __name__ == "__main__":
    unittest.main()
