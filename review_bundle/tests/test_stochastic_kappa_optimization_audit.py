from __future__ import annotations

import unittest

import numpy as np
import torch
from torch.distributions import Normal

from cert_runtime.generator_sac import GeneratorSACConfig, PersistentGeneratorSAC
from cert_runtime.optimization_diagnostics import (
    affine_scale_entropy_audit,
    entropy_decomposition,
    goal_projection_metrics,
)
from cert_runtime.task_authority import BestInGeneratorGoalOracle, action_from_eta
from envs.certified_uav import make_random_persistent_uav_env
from envs.certified_uav.persistent_task import PersistentMissionMode
from envs.certified_uav.state import UAVPhysicalState


class EntropyDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.mean = torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float32)
        self.std = torch.tensor([[0.8, 0.9, 1.1]], dtype=torch.float32)
        self.distribution = Normal(self.mean, self.std)
        self.u = torch.tensor([[0.4, -0.2, 0.1]], dtype=torch.float32)
        self.G = torch.diag_embed(torch.tensor([[0.04, 0.03, 0.02]], dtype=torch.float32))

    def test_actor_entropy_terms_decompose_exactly(self):
        terms = entropy_decomposition(self.distribution, self.u, self.G)
        self.assertTrue(torch.allclose(terms.normalized_log_prob, terms.normal_term + terms.negative_tanh_log_jacobian_term))
        self.assertTrue(torch.allclose(terms.physical_log_prob, terms.normalized_log_prob + terms.negative_log_det_G_term))

    def test_normalized_log_prob_excludes_only_log_det_G(self):
        terms = entropy_decomposition(self.distribution, self.u, self.G)
        self.assertTrue(torch.allclose(terms.physical_log_prob - terms.normalized_log_prob, -terms.log_det_G))

    def test_physical_log_prob_includes_log_det_G(self):
        terms = entropy_decomposition(self.distribution, self.u, self.G)
        expected = self.distribution.log_prob(self.u).sum(-1) - terms.tanh_log_jacobian - torch.linalg.slogdet(self.G).logabsdet
        self.assertTrue(torch.allclose(terms.physical_log_prob, expected))

    def test_affine_G_scaling_shifts_physical_log_prob_correctly(self):
        base = entropy_decomposition(self.distribution, self.u, self.G)
        doubled = entropy_decomposition(self.distribution, self.u, 2.0 * self.G)
        expected = torch.tensor([3.0 * np.log(2.0)], dtype=base.physical_log_prob.dtype)
        self.assertTrue(torch.allclose(base.physical_log_prob - doubled.physical_log_prob, expected, atol=1e-6))

    def test_affine_G_scaling_does_not_shift_normalized_log_prob(self):
        base = entropy_decomposition(self.distribution, self.u, self.G)
        doubled = entropy_decomposition(self.distribution, self.u, 2.0 * self.G)
        self.assertTrue(torch.allclose(base.normalized_log_prob, doubled.normalized_log_prob))

    def test_affine_scale_audit_records_alpha_gradient_direction(self):
        result = affine_scale_entropy_audit(self.distribution, self.u, self.G, -3.0, torch.tensor(0.2))
        self.assertEqual(set(result), {"0.5G", "1G", "2G"})
        self.assertAlmostEqual(result["0.5G"]["normalized_log_prob"], result["2G"]["normalized_log_prob"], places=6)


class StochasticKappaAndOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environment = make_random_persistent_uav_env("random_persistent_open.json", seed=1)
        cls.environment.reset(seed=1)

    def setUp(self):
        self.environment.reset(seed=1)

    def _set_charging_witness(self):
        self.environment.plant.state = UAVPhysicalState(
            np.array((0.3266141817168801, 0.4999997988646291, 1.0)),
            np.array((-0.04192432818426801, 1.0098367305728918e-7, 0.0)),
            4.1719195663088575,
            0.0,
        )
        self.environment.plant.failure_reason = None
        self.environment.plant.last_lidar = self.environment.plant.lidar_model.measure(
            self.environment.plant.state,
            self.environment.plant.world,
            self.environment.plant.np_random,
        )
        self.environment.task_env.mode = PersistentMissionMode.CHARGING_RL
        self.environment.task_env.phase = self.environment.task_env.mode
        self.environment._context_cache_key = None

    def test_invalid_kappa_failure_category_is_recorded(self):
        self.environment.plant.state = UAVPhysicalState(
            np.array((0.2851090968, 0.5, 1.0)), np.array((-0.0377, 0.0, 0.0)), 5.3, 0.0
        )
        self.environment.task_env.mode = PersistentMissionMode.BACKUP_RECOVERY
        self.environment.task_env.phase = self.environment.task_env.mode
        self.environment._context_cache_key = None
        context = self.environment._refresh_context()
        self.assertFalse(context["certificate_valid"])
        self.assertIn(context["kappa_validation_failure_category"], {"NO_CELL", "CELL_CONTAINMENT"})

    def test_stochastic_kappa_replay_is_reproducible(self):
        traces = []
        for _ in range(2):
            self.environment.reset(seed=1)
            self._set_charging_witness()
            trace = []
            for _ in range(12):
                _, _, terminated, truncated, info = self.environment.step(np.zeros(3))
                self.assertFalse(terminated or truncated)
                self.assertNotEqual(info.get("backup_reason"), "KAPPA_CERTIFICATE_INVALID")
                trace.append(np.concatenate((self.environment.plant.state.position, self.environment.plant.state.velocity)))
            traces.append(np.stack(trace))
        self.assertTrue(np.allclose(traces[0], traces[1]))

    def test_kappa_remains_valid_across_random_recovery_boundary(self):
        rng = np.random.default_rng(7)
        for _ in range(5):
            self._set_charging_witness()
            self.environment.plant.state.velocity = np.array((-0.04, 0.0, 0.0)) + rng.uniform(-0.001, 0.001, 3)
            for _ in range(8):
                _, _, terminated, truncated, info = self.environment.step(np.zeros(3))
                self.assertFalse(terminated or truncated)
                self.assertIsNone(info["action_context"].get("kappa_validation_failure_category"))

    def test_terminal_hold_certificate_binds_nonzero_hold(self):
        self._set_charging_witness()
        context = self.environment._refresh_context()
        self.assertTrue(context["certificate_valid"])
        self.assertGreater(np.linalg.norm(context["kappa"]), 0.0)
        self.assertEqual(context["recovery_hash"], self.environment.atlas.terminal_recovery_certificate.certificate_hash)

    def test_actor_goal_projection_metric(self):
        result = goal_projection_metrics(
            np.zeros(3), np.zeros(3), np.array((1.0, 0.0, 0.0)),
            np.array((0.1, 0.0, 0.0)), np.zeros(3), np.array((0.2, 0.0, 0.0)), 0.2,
        )
        self.assertGreater(result["actor_goal_projection"], 0.0)

    def test_oracle_gap_metric(self):
        result = goal_projection_metrics(
            np.zeros(3), np.zeros(3), np.array((1.0, 0.0, 0.0)),
            np.zeros(3), np.zeros(3), np.array((0.2, 0.0, 0.0)), 0.2,
        )
        self.assertGreater(result["oracle_gap"], 0.0)

    def test_goal_features_reach_actor(self):
        observation, _ = self.environment.reset(seed=1)
        agent = PersistentGeneratorSAC(observation.size, GeneratorSACConfig(hidden_dim=32), seed=3)
        first = torch.as_tensor(observation, dtype=torch.float32)
        goal_slice = self.environment.task_env.observation_layout["goal_delta"]
        second = first.clone()
        second[goal_slice] *= -1.0
        with torch.no_grad():
            first_mean = agent.actor.distribution(first).mean
            second_mean = agent.actor.distribution(second).mean
        self.assertFalse(torch.allclose(first_mean, second_mean))

    def test_goal_features_reach_critic(self):
        observation, _ = self.environment.reset(seed=1)
        agent = PersistentGeneratorSAC(observation.size, GeneratorSACConfig(hidden_dim=32), seed=3)
        first = torch.as_tensor(observation, dtype=torch.float32)
        goal_slice = self.environment.task_env.observation_layout["goal_delta"]
        second = first.clone()
        second[goal_slice] *= -1.0
        action = torch.zeros(3)
        with torch.no_grad():
            first_q = agent.critic_1(first, action)
            second_q = agent.critic_1(second, action)
        self.assertFalse(torch.allclose(first_q, second_q))

    def test_reward_decomposition_sums_to_total_reward(self):
        _, reward, terminated, truncated, info = self.environment.step(np.zeros(3))
        self.assertFalse(terminated or truncated)
        self.assertAlmostEqual(sum(info["reward_components"].values()), reward, places=7)

    def test_oracle_immediate_reward_vs_opposite_in_open_state(self):
        context = self.environment._refresh_context()
        state = self.environment.plant.state.copy()
        goal = self.environment.task_env.manager.current_task.goal_position.copy()
        oracle_eta = BestInGeneratorGoalOracle().select_eta(state, goal, context["c"], context["G"], self.environment.plant.config.dt)
        oracle = action_from_eta(context["c"], context["G"], oracle_eta)
        opposite = action_from_eta(context["c"], context["G"], -oracle_eta)
        dt = self.environment.plant.config.dt
        oracle_position = state.position + state.velocity * dt + 0.5 * oracle * dt * dt
        opposite_position = state.position + state.velocity * dt + 0.5 * opposite * dt * dt
        oracle_progress = np.linalg.norm(goal - state.position) - np.linalg.norm(goal - oracle_position)
        opposite_progress = np.linalg.norm(goal - state.position) - np.linalg.norm(goal - opposite_position)
        self.assertGreaterEqual(oracle_progress, opposite_progress)

    def test_critic_action_ordering_diagnostic(self):
        observation, _ = self.environment.reset(seed=1)
        agent = PersistentGeneratorSAC(observation.size, GeneratorSACConfig(hidden_dim=32), seed=3)
        tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        actions = torch.stack((torch.zeros(3), torch.ones(3) * 0.01, -torch.ones(3) * 0.01))
        with torch.no_grad():
            values = agent.critic_1(tensor.repeat(3, 1), actions)
        self.assertEqual(tuple(values.shape), (3,))
        self.assertTrue(torch.isfinite(values).all())

    def test_task_switch_next_observation_uses_new_goal(self):
        task = self.environment.task_env.manager.current_task
        old_goal = self.environment.plant.state.position.copy()
        task.goal_position = old_goal
        _, _, terminated, truncated, info = self.environment.step(np.zeros(3))
        self.assertFalse(terminated or truncated)
        self.assertTrue(info["task_completed_now"])
        new_goal = self.environment.task_env.manager.current_task.goal_position
        self.assertFalse(np.allclose(old_goal, new_goal))
        observation = self.environment.task_env.build_observation(
            self.environment.runtime._map_encoding(), self.environment.runtime._corridor_encoding()
        )
        goal_slice = self.environment.task_env.observation_layout["goal_delta"]
        expected = (new_goal - self.environment.plant.state.position) / self.environment.plant.config.world_size
        self.assertTrue(np.allclose(observation[goal_slice], expected.astype(np.float32)))


if __name__ == "__main__":
    unittest.main()
