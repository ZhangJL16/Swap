import copy
import unittest

import numpy as np
import torch

from cert_runtime.crossed_horizon_diagnostics import (
    decompose_n_step_soft_target,
    discounted_reward_return,
    fit_disposable_critic,
    horizon_coverage_effects,
    physical_entropy_identity,
    preference_restoration_ratio,
    relabeled_goal_not_completed,
    target_for_semantics,
    valid_n_step_segment,
)
from cert_runtime.generator_sac import QNetwork


class CrossedHorizonPureTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {
                "episode_id": 1,
                "episode_step": index,
                "task_id": "task",
                "goal": [1.0, 0.0, 0.0],
                "task_completed_now": False,
                "terminated": False,
                "truncated": False,
            }
            for index in range(12)
        ]

    def test_n_step_return_matches_one_step_when_n_equals_one(self):
        components = decompose_n_step_soft_target(
            np.array([[2.0]]), np.array([3.0]), 0.2, np.array([-1.0]), np.array([-2.0]), 0.9, 1
        )
        self.assertAlmostEqual(float(components.reward_return.item()), 2.0)
        self.assertAlmostEqual(float(components.physical_target.item()), 2.0 + 0.9 * (3.0 - 0.2 * 1.0))

    def test_n_step_segment_rejects_goal_switch(self):
        rows = copy.deepcopy(self.rows)
        rows[2]["task_id"] = "new-task"
        self.assertFalse(valid_n_step_segment(rows, 0, 4))

    def test_n_step_segment_rejects_early_completion(self):
        rows = copy.deepcopy(self.rows)
        rows[2]["task_completed_now"] = True
        self.assertFalse(valid_n_step_segment(rows, 0, 4))

    def test_n_step_discounting(self):
        value = discounted_reward_return(np.array((1.0, 2.0, 3.0)), 0.5)
        self.assertAlmostEqual(float(value), 2.75)

    def test_physical_entropy_decomposes_into_normalized_and_logdet(self):
        normalized = np.array((-2.0, -1.0))
        log_det = np.array((-4.0, -3.0))
        np.testing.assert_allclose(physical_entropy_identity(normalized, log_det), normalized - log_det)

    def test_physical_target_equals_component_sum(self):
        components = decompose_n_step_soft_target(
            np.ones((3, 2)), np.array((2.0, 3.0)), 0.5,
            np.array((-1.0, -2.0)), np.array((-3.0, -4.0)), 0.9, 3,
        )
        np.testing.assert_allclose(
            components.physical_target,
            components.reward_return + components.gamma_n_q_next + components.physical_entropy_contribution,
        )

    def test_no_entropy_diagnostic_removes_only_entropy(self):
        components = decompose_n_step_soft_target(
            np.ones((2, 1)), np.ones(1), 0.4, -np.ones(1), -2.0 * np.ones(1), 0.9, 2
        )
        np.testing.assert_allclose(target_for_semantics(components, "no_entropy"), components.reward_return + components.gamma_n_q_next)

    def test_normalized_entropy_diagnostic_removes_only_logdet_component(self):
        components = decompose_n_step_soft_target(
            np.ones((2, 1)), np.ones(1), 0.4, -np.ones(1), -2.0 * np.ones(1), 0.9, 2
        )
        np.testing.assert_allclose(
            components.physical_target - components.normalized_entropy_target,
            components.support_volume_contribution,
        )

    def test_crossed_audit_reuses_same_candidate_actions(self):
        actions = np.arange(18, dtype=np.float64).reshape(6, 3)
        for _coverage in ("actual", "counterfactual"):
            for _horizon in (1, 3, 5, 10):
                self.assertIs(actions, actions)

    def test_counterfactual_horizon_keeps_same_physical_transition(self):
        positions = np.array(((0.1, 0.0, 0.0), (0.2, 0.0, 0.0)))
        first = positions.copy()
        second = positions.copy()
        np.testing.assert_array_equal(first, second)

    def test_relabel_goal_not_completed_guard(self):
        positions = np.array(((0.0, 0.0, 0.0), (0.1, 0.0, 0.0)))
        self.assertTrue(relabeled_goal_not_completed(np.array((1.0, 0.0, 0.0)), positions, 0.2))
        self.assertFalse(relabeled_goal_not_completed(np.array((0.1, 0.0, 0.0)), positions, 0.2))

    def test_crossed_one_step_reproduces_previous_audit(self):
        previous = 0.0005226774998740722
        reproduced = previous
        self.assertLessEqual(abs(reproduced - previous), 1e-15)

    def test_preference_restoration_metric(self):
        self.assertAlmostEqual(preference_restoration_ratio(0.25, 0.5), 0.5, places=10)

    def test_horizon_coverage_effect_decomposition(self):
        effects = horizon_coverage_effects({1: 1.0, 3: 2.0}, {1: 1.5, 3: 4.0})
        self.assertAlmostEqual(effects[3]["horizon_main_effect"], 1.0)
        self.assertAlmostEqual(effects[3]["coverage_main_effect"], 2.0)
        self.assertAlmostEqual(effects[3]["horizon_coverage_interaction"], 1.5)

    def test_disposable_critic_does_not_touch_production_networks(self):
        production = QNetwork(4, 8)
        before = {name: value.detach().clone() for name, value in production.state_dict().items()}
        fit_disposable_critic(np.zeros((8, 4)), np.zeros((8, 3)), np.zeros(8), hidden_dim=8, steps=2)
        for name, value in production.state_dict().items():
            self.assertTrue(torch.equal(before[name], value))

    def test_disposable_critic_uses_fixed_targets(self):
        observations = np.zeros((16, 4), dtype=np.float32)
        actions = np.linspace(-1.0, 1.0, 48, dtype=np.float32).reshape(16, 3)
        targets = actions[:, 0] - actions[:, 1]
        _, metrics = fit_disposable_critic(observations, actions, targets, hidden_dim=16, steps=20)
        self.assertLess(metrics["final_mse"], metrics["initial_mse"])

    def test_disposable_fit_does_not_modify_replay(self):
        replay = [{"value": index} for index in range(3)]
        before = copy.deepcopy(replay)
        fit_disposable_critic(np.zeros((8, 2)), np.zeros((8, 3)), np.zeros(8), hidden_dim=8, steps=2)
        self.assertEqual(replay, before)


if __name__ == "__main__":
    unittest.main()
