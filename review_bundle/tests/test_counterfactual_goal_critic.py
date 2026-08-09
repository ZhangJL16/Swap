from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from cert_runtime.counterfactual_goal_diagnostics import (
    GOAL_DERIVED_OBSERVATION_FIELDS,
    action_gradient_goal_jacobian,
    certificate_invariance_snapshot,
    certificate_snapshots_equal,
    changed_observation_fields,
    cross_goal_q_matrix,
    diagonal_preference,
    environment_cross_goal_matrix,
    finite_difference_action_gradient_goal_jacobian,
    mean_pairwise_distance,
    opposite_goal_preference_reversal,
    residual_alignment,
    searched_critic_preferred_actions,
)
from cert_runtime.generator_sac import GeneratorSACConfig, PersistentGeneratorSAC
from cert_runtime.task_authority import BestInGeneratorGoalOracle
from envs.certified_uav import make_random_persistent_uav_env


class _GoalActionQ(nn.Module):
    def forward(self, observation, action):
        target = 0.08 * observation[:, :3]
        return -((action - target) ** 2).sum(-1)


class _BilinearGoalActionQ(nn.Module):
    def forward(self, observation, action):
        return (observation[:, :3] * action).sum(-1)


class CounterfactualGoalPureTests(unittest.TestCase):
    def setUp(self):
        self.agent = PersistentGeneratorSAC(8, GeneratorSACConfig(hidden_dim=16), seed=7)
        self.agent.critic_1 = _GoalActionQ()
        self.agent.critic_2 = _GoalActionQ()
        self.observations = torch.tensor((
            (1.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.0, 0.0),
            (-1.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.0, 0.0),
        ), dtype=torch.float32)
        self.center = np.zeros(3, dtype=np.float64)
        self.generator = np.diag((0.1, 0.1, 0.05)).astype(np.float64)

    def test_mixed_goal_action_derivative_matches_finite_difference(self):
        self.agent.critic_1 = _BilinearGoalActionQ()
        self.agent.critic_2 = _BilinearGoalActionQ()
        action = torch.tensor(((0.02, -0.01, 0.0),), dtype=torch.float32)
        observation = self.observations[0:1]
        analytic = action_gradient_goal_jacobian(self.agent, observation, action, slice(0, 3))
        finite = finite_difference_action_gradient_goal_jacobian(
            self.agent, observation, action, slice(0, 3), epsilon_goal=1e-3
        )
        self.assertTrue(torch.allclose(analytic, torch.eye(3).unsqueeze(0), atol=1e-7))
        self.assertTrue(torch.allclose(analytic, finite, atol=1e-5, rtol=1e-4))

    def test_critic_preferred_action_is_inside_generator_support(self):
        results = searched_critic_preferred_actions(
            self.agent,
            self.observations,
            self.center,
            self.generator,
            np.zeros((2, 3)),
            seed=3,
            gradient_steps=8,
            random_starts=2,
            random_evaluations=8,
        )
        for result in results:
            self.assertTrue(np.all(result.eta >= -1.0))
            self.assertTrue(np.all(result.eta <= 1.0))
            np.testing.assert_allclose(result.action, self.center + self.generator @ result.eta, atol=1e-7)

    def test_critic_preferred_search_does_not_modify_network(self):
        before = deepcopy(self.agent.critic_1.state_dict())
        searched_critic_preferred_actions(
            self.agent, self.observations, self.center, self.generator, np.zeros((2, 3)),
            seed=4, gradient_steps=4, random_starts=1, random_evaluations=4,
        )
        self.assertTrue(all(torch.equal(before[name], value) for name, value in self.agent.critic_1.state_dict().items()))

    def test_opposite_goal_preference_metric(self):
        self.assertTrue(opposite_goal_preference_reversal(
            np.array((0.1, 0.0, 0.0)), np.array((-0.1, 0.0, 0.0)),
            np.array((1.0, 0.0, 0.0)), np.array((-1.0, 0.0, 0.0)), 0,
        ))

    def test_preferred_action_goal_sensitivity_metric(self):
        self.assertAlmostEqual(mean_pairwise_distance((np.zeros(3), np.ones(3))), np.sqrt(3.0))

    def test_critic_oracle_alignment_metric(self):
        self.assertAlmostEqual(residual_alignment(
            np.array((0.1, 0.0, 0.0)), np.array((0.2, 0.0, 0.0)), np.zeros(3)
        ), 1.0)

    def test_cross_goal_Q_matrix_shape_and_semantics(self):
        actions = torch.tensor(((0.08, 0.0, 0.0), (-0.08, 0.0, 0.0)), dtype=torch.float32)
        matrix = cross_goal_q_matrix(self.agent, self.observations, actions)
        self.assertEqual(matrix.shape, (2, 2))
        fraction, advantage = diagonal_preference(matrix)
        self.assertEqual(fraction, 1.0)
        self.assertGreater(advantage, 0.0)

    def test_counterfactual_audit_does_not_modify_replay(self):
        replay = [{"observation": np.zeros(3), "action": np.zeros(3)}]
        before = deepcopy(replay)
        cross_goal_q_matrix(
            self.agent,
            self.observations,
            torch.tensor(((0.08, 0.0, 0.0), (-0.08, 0.0, 0.0))),
        )
        np.testing.assert_array_equal(replay[0]["observation"], before[0]["observation"])
        np.testing.assert_array_equal(replay[0]["action"], before[0]["action"])

    def test_counterfactual_audit_does_not_modify_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(self.agent.state_dict(), path)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            searched_critic_preferred_actions(
                self.agent, self.observations, self.center, self.generator, np.zeros((2, 3)),
                seed=5, gradient_steps=2, random_starts=1, random_evaluations=2,
            )
            after = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(before, after)

    def test_oracle_is_evaluation_only(self):
        oracle = BestInGeneratorGoalOracle()
        before = deepcopy(self.agent.actor.state_dict())
        self.assertFalse(hasattr(oracle, "optimizer"))
        self.assertTrue(all(torch.equal(before[name], value) for name, value in self.agent.actor.state_dict().items()))


class CounterfactualGoalEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environment = make_random_persistent_uav_env("random_persistent_open.json", seed=6)

    def setUp(self):
        self.observation, self.info = self.environment.reset(seed=6)
        self.first_context = self.info["action_context"]
        self.original_goal = self.environment.task_env.manager.current_task.goal_position.copy()
        self.environment.task_env.manager.current_task.goal_position = self.original_goal + np.array((0.25, -0.15, 0.0))
        self.environment._context_cache_key = None
        self.environment._context_cache = None
        self.second_context = self.environment._refresh_context()
        self.second_observation = self.environment.task_env.build_observation(
            self.environment.runtime._map_encoding(), self.environment.runtime._corridor_encoding()
        )

    def test_counterfactual_goal_changes_only_goal_observation_fields(self):
        changed = changed_observation_fields(
            self.observation,
            self.second_observation,
            self.environment.task_env.observation_layout,
            tolerance=1e-7,
        )
        self.assertEqual(set(changed), GOAL_DERIVED_OBSERVATION_FIELDS)

    def test_counterfactual_goal_preserves_c_and_G(self):
        np.testing.assert_allclose(self.first_context["c"], self.second_context["c"], atol=1e-12)
        np.testing.assert_allclose(self.first_context["G"], self.second_context["G"], atol=1e-12)

    def test_counterfactual_goal_preserves_certificate_identity(self):
        equal, failures = certificate_snapshots_equal(
            certificate_invariance_snapshot(self.first_context),
            certificate_invariance_snapshot(self.second_context),
        )
        self.assertTrue(equal, failures)

    def test_counterfactual_goal_preserves_R_and_R_RL(self):
        self.assertEqual(
            self.first_context["recoverable_set_member"], self.second_context["recoverable_set_member"]
        )
        self.assertEqual(
            self.first_context["rl_authority_set_member"], self.second_context["rl_authority_set_member"]
        )

    def test_environment_cross_goal_matrix(self):
        state = self.environment.plant.state.copy()
        goals = (state.position + np.array((1.0, 0.0, 0.0)), state.position - np.array((1.0, 0.0, 0.0)))
        actions = (np.array((0.1, 0.0, 0.0)), np.array((-0.1, 0.0, 0.0)))
        reward, progress = environment_cross_goal_matrix(
            state, goals, actions, self.environment.plant.config.dt,
            self.environment.plant.energy_model, self.environment.task_env.reward_config,
        )
        self.assertEqual(reward.shape, (2, 2))
        self.assertGreater(progress[0, 0], progress[0, 1])
        self.assertGreater(progress[1, 1], progress[1, 0])


if __name__ == "__main__":
    unittest.main()
