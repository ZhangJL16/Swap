from __future__ import annotations

from copy import deepcopy
import unittest

import numpy as np
import torch
from torch import nn

from cert_runtime.actor_gradient_diagnostics import (
    action_to_latent_gradient,
    actor_goal_jacobians,
    actor_gradient_decomposition,
    critic_action_column_statistics,
    critic_action_gradient,
    critic_goal_jacobian,
    directional_finite_difference,
    gradient_cosine,
)
from cert_runtime.generator_sac import GeneratorSACConfig, PersistentGeneratorSAC
from envs.certified_uav import make_random_persistent_uav_env
from scripts.audit_actor_gradient_learning import _aggregate


class _SmoothQ(nn.Module):
    def __init__(self, target=(0.1, -0.05, 0.02), goal_index=0):
        super().__init__()
        self.register_buffer("target", torch.tensor(target, dtype=torch.float32))
        self.goal_index = goal_index

    def forward(self, observation, action):
        return -((action - self.target) ** 2).sum(-1) + 0.2 * observation[:, self.goal_index]


class ActorGradientPureTests(unittest.TestCase):
    def setUp(self):
        self.agent = PersistentGeneratorSAC(8, GeneratorSACConfig(hidden_dim=16), seed=2)
        self.agent.critic_1 = _SmoothQ()
        self.agent.critic_2 = _SmoothQ()
        self.observation = torch.tensor([[0.2, -0.1, 0.3, 0.4, -0.2, 0.1, 0.0, 0.5]])
        self.action = torch.tensor([[0.02, -0.01, 0.0]])
        self.oracle = torch.tensor([[0.08, -0.04, 0.015]])
        self.center = torch.zeros((1, 3))
        self.generator = torch.diag_embed(torch.tensor([[0.04, 0.03, 0.02]]))

    def test_critic_action_gradient_matches_finite_difference(self):
        _, gradient = critic_action_gradient(self.agent, self.observation, self.action)
        direction = self.oracle - self.action
        direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        autograd_value = (gradient * direction).sum(-1)
        finite_difference = directional_finite_difference(
            self.agent, self.observation, self.action, direction, epsilon_action=1e-3
        )
        self.assertTrue(torch.allclose(autograd_value, finite_difference, atol=1e-4, rtol=1e-3))

    def test_Q_directional_derivative_toward_oracle_metric(self):
        _, gradient = critic_action_gradient(self.agent, self.observation, self.action)
        direction = self.oracle - self.action
        direction /= torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        self.assertGreater(float((gradient * direction).sum()), 0.0)

    def test_action_to_latent_gradient_chain_rule(self):
        u = torch.tensor([[0.2, -0.3, 0.1]], requires_grad=True)
        action = self.center + torch.bmm(self.generator, torch.tanh(u).unsqueeze(-1)).squeeze(-1)
        value = self.agent.critic_1(self.observation, action)
        direct = torch.autograd.grad(value.sum(), u)[0]
        _, gradient_action = critic_action_gradient(self.agent, self.observation, action.detach())
        result = action_to_latent_gradient(gradient_action, self.generator, u.detach())
        self.assertTrue(torch.allclose(direct, result.grad_u, atol=1e-7))

    def test_actor_Q_and_entropy_gradients_sum_to_total(self):
        result = actor_gradient_decomposition(
            self.agent,
            self.observation,
            self.center,
            self.generator,
            torch.zeros((1, 3)),
        )
        self.assertLess(result["gradient_sum_consistency_error"], 1e-10)

    def test_gradient_cosine_metric(self):
        self.assertAlmostEqual(gradient_cosine(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])), 0.0)
        self.assertAlmostEqual(gradient_cosine(torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])), -1.0)

    def test_frozen_critic_Q_only_updates_actor_only(self):
        actor = deepcopy(self.agent.actor)
        critic_before = [parameter.detach().clone() for critic in (self.agent.critic_1, self.agent.critic_2) for parameter in critic.parameters()]
        actor_before = [parameter.detach().clone() for parameter in actor.parameters()]
        optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
        mean = actor.distribution(self.observation).mean
        action = self.center + torch.bmm(self.generator, torch.tanh(mean).unsqueeze(-1)).squeeze(-1)
        loss = -torch.minimum(self.agent.critic_1(self.observation, action), self.agent.critic_2(self.observation, action)).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        critic_after = [parameter.detach().clone() for critic in (self.agent.critic_1, self.agent.critic_2) for parameter in critic.parameters()]
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(critic_before, critic_after)))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(actor_before, actor.parameters())))

    def test_Q_only_does_not_update_critic(self):
        self.test_frozen_critic_Q_only_updates_actor_only()

    def test_goal_jacobian_autograd_matches_finite_difference(self):
        _, action_jacobian = actor_goal_jacobians(
            self.agent, self.observation, self.center, self.generator, slice(0, 3)
        )
        epsilon = 1e-3
        columns = []
        for axis in range(3):
            plus = self.observation.clone(); minus = self.observation.clone()
            plus[:, axis] += epsilon; minus[:, axis] -= epsilon
            with torch.no_grad():
                plus_action = self.center + torch.bmm(self.generator, torch.tanh(self.agent.actor.distribution(plus).mean).unsqueeze(-1)).squeeze(-1)
                minus_action = self.center + torch.bmm(self.generator, torch.tanh(self.agent.actor.distribution(minus).mean).unsqueeze(-1)).squeeze(-1)
            columns.append(((plus_action - minus_action) / (2 * epsilon))[0])
        finite = torch.stack(columns, dim=1)
        self.assertTrue(torch.allclose(action_jacobian[0], finite, atol=1e-5, rtol=2e-3))

    def test_actor_goal_jacobian_uses_goal_features_only(self):
        mean_jacobian, action_jacobian = actor_goal_jacobians(
            self.agent, self.observation, self.center, self.generator, slice(0, 3)
        )
        self.assertEqual(tuple(mean_jacobian.shape), (1, 3, 3))
        self.assertEqual(tuple(action_jacobian.shape), (1, 3, 3))

    def test_critic_goal_jacobian_metric(self):
        jacobian = critic_goal_jacobian(self.agent, self.observation, self.action, slice(0, 3))
        self.assertAlmostEqual(float(jacobian[0, 0]), 0.2, places=6)
        self.assertTrue(torch.allclose(jacobian[0, 1:], torch.zeros(2)))

    def test_critic_action_column_extraction(self):
        agent = PersistentGeneratorSAC(8, GeneratorSACConfig(hidden_dim=16), seed=2)
        result = critic_action_column_statistics(agent.critic_1, 8)
        self.assertEqual(len(result["per_action_axis_L2"]), 3)
        self.assertGreater(result["action_column_L2"], 0.0)

    def test_actor_gradient_audit_does_not_modify_checkpoint(self):
        before = {name: value.detach().clone() for name, value in self.agent.actor.state_dict().items()}
        actor_gradient_decomposition(
            self.agent, self.observation, self.center, self.generator, torch.zeros((1, 3))
        )
        self.assertTrue(all(torch.equal(before[name], value) for name, value in self.agent.actor.state_dict().items()))

    def test_mixed_diagnosis_blocks_candidate_when_goal_sensitivity_decreases(self):
        checkpoints = []
        frozen = []
        for cosine in (-0.5, 0.4, -0.2):
            checkpoints.append({
                "CRITIC_LOCAL_GRADIENT": {
                    "fraction_directional_derivative_toward_oracle_gt_0": 0.9,
                },
                "ACTOR_GRADIENT_DECOMPOSITION": {
                    "Q_gradient_norm": 1.0,
                    "entropy_gradient_norm": 2.0,
                    "Q_entropy_cosine": cosine,
                },
                "GRADIENT_TRANSMISSION": {"total_transmission_ratio": {"mean": 0.1}},
                "GOAL_CONDITIONING": {
                    "trained_action_goal_jacobian_norm": 0.8,
                    "untrained_action_goal_jacobian_norm": 1.0,
                    "critic_goal_jacobian_norm": 2.0,
                },
            })
            frozen.append({
                "trajectories": {
                    "Q_ONLY": [
                        {"Q_actor": 0.0, "actor_goal_projection": 0.0, "oracle_gap": 1.0, "action_goal_jacobian_norm": 1.0},
                        {"Q_actor": 1.0, "actor_goal_projection": 1.0, "oracle_gap": 0.0, "action_goal_jacobian_norm": 0.5},
                    ],
                    "CURRENT_ACTOR_OBJECTIVE": [
                        {"Q_actor": 0.0, "actor_goal_projection": 0.0, "oracle_gap": 1.0, "action_goal_jacobian_norm": 1.0},
                        {"Q_actor": -0.1, "actor_goal_projection": 0.0, "oracle_gap": 1.1, "action_goal_jacobian_norm": 0.8},
                    ],
                },
            })
        result = _aggregate(checkpoints, frozen)
        self.assertEqual(result["PRIMARY_CLASSIFICATION"], "MIXED")
        self.assertEqual(result["ACTOR_GRADIENT_LEARNING_GATE"], "MARGINAL")


class ActorGradientEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environment = make_random_persistent_uav_env("random_persistent_open.json", seed=4)

    def test_goal_change_does_not_change_c_or_G(self):
        _, info = self.environment.reset(seed=4)
        first = info["action_context"]
        self.environment.task_env.manager.current_task.goal_position += np.array((0.2, -0.1, 0.0))
        self.environment._context_cache_key = None
        second = self.environment._refresh_context()
        np.testing.assert_allclose(first["c"], second["c"])
        np.testing.assert_allclose(first["G"], second["G"])


if __name__ == "__main__":
    unittest.main()
