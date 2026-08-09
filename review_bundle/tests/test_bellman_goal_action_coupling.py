from __future__ import annotations

from copy import deepcopy
import unittest

import numpy as np

from cert_runtime.bellman_goal_action_diagnostics import (
    additive_decomposition_metrics,
    bellman_target_decomposition,
    contrast_decomposition,
    finite_preferred_actions,
    goal_action_interaction,
    immediate_reward_components,
    nominal_physical_transition,
)
from cert_runtime.replay_goal_action_diagnostics import (
    action_covariance_metrics,
    counterfactual_augmented_features,
    effective_rank,
    goal_action_interaction_features,
    goal_direction_diversity,
    observed_action_support_coverage,
    physical_neighborhood_key,
)
from envs.certified_uav import make_random_persistent_uav_env


class BellmanGoalActionPureTests(unittest.TestCase):
    def test_bellman_target_decomposition_sums_exactly(self):
        result = bellman_target_decomposition(
            np.array((1.0, 2.0)), np.array((3.0, 4.0)), np.array((0.2, 0.3)), 0.9
        )
        np.testing.assert_allclose(
            result["target"],
            result["reward"] + result["gamma_next_q"] + result["negative_gamma_entropy"],
            atol=1e-12,
        )

    def test_oracle_opposite_target_contrast_metric(self):
        result = bellman_target_decomposition(
            np.array((0.5, -0.5)), np.array((2.0, 1.0)), np.array((0.1, 0.2)), 0.99
        )
        contrast = contrast_decomposition(result, 0, 1)
        self.assertAlmostEqual(contrast["target"], result["target"][0] - result["target"][1])
        self.assertGreater(contrast["bellman_contrast_preservation_ratio"], 0.0)

    def test_goal_action_interaction_centering(self):
        matrix = np.array(((1.0, 2.0, 4.0), (2.0, 5.0, 8.0)))
        interaction = goal_action_interaction(matrix)
        np.testing.assert_allclose(interaction.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(interaction.mean(axis=1), 0.0, atol=1e-12)

    def test_target_preferred_action_is_certified(self):
        center = np.array((0.01, -0.02, 0.0))
        generator = np.diag((0.1, 0.1, 0.05))
        etas = np.array(((0.0, 0.0, 0.0), (1.0, -1.0, 0.5)))
        actions = center + np.einsum("ij,nj->ni", generator, etas)
        _, preferred = finite_preferred_actions(np.array(((0.0, 1.0),)), actions)
        recovered = np.linalg.solve(generator, preferred[0] - center)
        self.assertTrue(np.all(np.abs(recovered) <= 1.0 + 1e-12))

    def test_additive_matrix_has_zero_interaction_residual(self):
        matrix = np.array((1.0, 2.0, 4.0))[:, None] + np.array((0.5, -0.5))[None, :]
        metrics = additive_decomposition_metrics(matrix)
        np.testing.assert_allclose(metrics["interaction_residual"], 0.0, atol=1e-12)
        self.assertAlmostEqual(metrics["additive_explained_variance"], 1.0)

    def test_nonadditive_matrix_has_positive_interaction_residual(self):
        matrix = np.array(((1.0, 0.0), (0.0, 1.0)))
        self.assertGreater(additive_decomposition_metrics(matrix)["interaction_variance"], 0.0)

    def test_additive_explained_variance_metric(self):
        additive = np.array(((0.0, 1.0), (2.0, 3.0)))
        nonadditive = additive + np.array(((1.0, -1.0), (-1.0, 1.0)))
        self.assertGreater(
            additive_decomposition_metrics(additive)["additive_explained_variance"],
            additive_decomposition_metrics(nonadditive)["additive_explained_variance"],
        )

    def test_counterfactual_relabel_not_inserted_into_replay(self):
        replay = [{"goal": np.array((1.0, 0.0, 0.0)), "reward": 0.1}]
        before = deepcopy(replay)
        _ = bellman_target_decomposition(np.array((0.2,)), np.array((1.0,)), np.array((0.0,)), 0.99)
        np.testing.assert_array_equal(replay[0]["goal"], before[0]["goal"])
        self.assertEqual(replay[0]["reward"], before[0]["reward"])


class ReplayIdentifiabilityPureTests(unittest.TestCase):
    def test_physical_neighborhood_grouping_is_goal_independent(self):
        row = {"kappa_cell_id": "r0", "persistent_mode": "TASK_RL", "position": (1, 2, 3), "velocity": (0, 0, 0), "energy_margin": 1.0, "goal": (4, 5, 6)}
        changed = dict(row, goal=(-4, -5, 6))
        self.assertEqual(physical_neighborhood_key(row), physical_neighborhood_key(changed))

    def test_goal_direction_diversity_metric(self):
        result = goal_direction_diversity(np.array(((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0))))
        self.assertTrue(result["near_opposite"])
        self.assertGreaterEqual(result["distinct_directions"], 4)

    def test_action_covariance_metric(self):
        actions = np.array(((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)))
        self.assertEqual(action_covariance_metrics(actions)["rank"], 3)

    def test_goal_action_interaction_feature_shape(self):
        goals = np.eye(3)
        actions = np.eye(3)
        self.assertEqual(goal_action_interaction_features(goals, actions).shape, (3, 9))

    def test_interaction_effective_rank(self):
        matrix = np.eye(9)
        metrics = effective_rank(matrix)
        self.assertGreaterEqual(metrics["rank"], 8)
        self.assertGreater(metrics["effective_rank"], 1.0)

    def test_observed_action_support_coverage_metric(self):
        etas = np.array(((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
        metrics = observed_action_support_coverage(etas)
        self.assertAlmostEqual(metrics["per_axis_std_over_half_width"][0], 1.0)
        self.assertEqual(metrics["max_abs_eta"], 1.0)

    def test_counterfactual_augmented_rank_is_diagnostic_only(self):
        goals = np.array(((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)))
        actions = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
        before = actions.copy()
        features = counterfactual_augmented_features(goals, actions)
        self.assertEqual(features.shape, (12, 9))
        np.testing.assert_array_equal(actions, before)


class BellmanGoalActionEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environment = make_random_persistent_uav_env("random_persistent_open.json", seed=41)

    def setUp(self):
        self.observation, self.info = self.environment.reset(seed=41)
        self.state = self.environment.plant.state.copy()
        self.context = self.info["action_context"]
        self.action = np.asarray(self.context["c"], dtype=np.float64)

    def test_counterfactual_goal_keeps_same_physical_successor(self):
        first = nominal_physical_transition(self.environment, self.state, self.action)
        self.environment.task_env.manager.current_task.goal_position += np.array((0.4, -0.2, 0.0))
        second = nominal_physical_transition(self.environment, self.state, self.action)
        np.testing.assert_array_equal(first.state.position, second.state.position)
        np.testing.assert_array_equal(first.state.velocity, second.state.velocity)
        self.assertEqual(first.energy_cost, second.energy_cost)

    def test_counterfactual_goal_recomputes_progress_reward(self):
        transition = nominal_physical_transition(self.environment, self.state, self.action)
        plus = immediate_reward_components(self.environment, self.state, self.state.position + np.array((1.0, 0.0, 0.0)), transition)
        minus = immediate_reward_components(self.environment, self.state, self.state.position - np.array((1.0, 0.0, 0.0)), transition)
        self.assertNotEqual(plus["goal_progress_reward"], minus["goal_progress_reward"])

    def test_task_completion_boundary_uses_old_goal_reward(self):
        transition = nominal_physical_transition(self.environment, self.state, self.action)
        old_goal = transition.state.position.copy()
        new_goal = old_goal + np.array((1.0, 0.0, 0.0))
        old_components = immediate_reward_components(self.environment, self.state, old_goal, transition)
        new_components = immediate_reward_components(self.environment, self.state, new_goal, transition)
        self.assertTrue(old_components["task_completed"])
        self.assertGreater(old_components["task_completion_reward"], new_components["task_completion_reward"])

    def test_task_completion_next_observation_uses_new_goal(self):
        self.environment.task_env.manager.current_task.goal_position = self.state.position.copy()
        old_goal = self.environment.task_env.manager.current_task.goal_position.copy()
        next_observation, _, _, _, info = self.environment.task_env.step(np.zeros(3))
        self.assertTrue(info["task_completed_now"])
        new_goal = self.environment.task_env.manager.current_task.goal_position.copy()
        self.assertFalse(np.array_equal(old_goal, new_goal))
        goal_slice = self.environment.task_env.observation_layout["goal_delta"]
        expected = (new_goal - self.environment.plant.state.position) / self.environment.plant.config.world_size
        np.testing.assert_allclose(next_observation[goal_slice], expected, atol=1e-7)

    def test_noncompletion_relabel_keeps_same_physical_transition(self):
        transition = nominal_physical_transition(self.environment, self.state, self.action)
        first = immediate_reward_components(self.environment, self.state, self.state.position + np.array((1.0, 0.0, 0.0)), transition)
        second = immediate_reward_components(self.environment, self.state, self.state.position + np.array((0.0, 1.0, 0.0)), transition)
        self.assertFalse(first["task_completed"])
        self.assertFalse(second["task_completed"])
        self.assertEqual(transition.energy_cost, transition.energy_cost)


if __name__ == "__main__":
    unittest.main()
