from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import torch

from cert_runtime.generator_sac import GeneratorSAC, GeneratorSACConfig, PersistentGeneratorSAC
from cert_runtime.goal_exposure import (
    GoalExposureAccumulator,
    batch_goal_diversity,
    goal_exposure_reset_boundary,
    goal_exposure_reset_seed,
    training_protocol_name,
)
from envs.certified_uav import make_random_persistent_uav_env
from scripts.persistent_generator_common import transition_from_cycle


class MultiGoalExposureProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.environment = make_random_persistent_uav_env("random_persistent_open.json", seed=0)

    def setUp(self) -> None:
        self.observation, self.reset_info = self.environment.reset(seed=0)

    def _collector_transition(self):
        context = self.reset_info["action_context"]
        actor_u = np.zeros(3, dtype=np.float64)
        next_observation, reward, terminated, truncated, info = self.environment.step(actor_u)
        next_context = None if terminated or truncated else self.environment._refresh_context()
        item = transition_from_cycle(
            self.observation,
            next_observation,
            actor_u,
            reward,
            terminated,
            truncated,
            0,
            context,
            next_context,
            info,
            collector_boundary=True,
        )
        return item, next_observation, next_context

    def test_goal_exposure_reset_samples_new_goal(self) -> None:
        first_goal = np.asarray(self.reset_info["sampled_goal"], dtype=np.float64)
        reset_seed = goal_exposure_reset_seed(0, 1)
        _, reset_info = self.environment.reset(seed=reset_seed)
        self.assertFalse(np.allclose(first_goal, reset_info["sampled_goal"]))

    def test_goal_exposure_reset_samples_certified_initial_state(self) -> None:
        reset_seed = goal_exposure_reset_seed(2, 3)
        _, reset_info = self.environment.reset(seed=reset_seed)
        context = reset_info["action_context"]
        self.assertTrue(context["certificate_valid"])
        self.assertTrue(context["recoverable_set_member"])
        self.assertTrue(context["rl_authority_set_member"])

    def test_exposure_reset_preserves_agent(self) -> None:
        agent = GeneratorSAC(self.observation.size, GeneratorSACConfig(hidden_dim=16), seed=4)
        before = [parameter.detach().clone() for parameter in agent.actor.parameters()]
        self.environment.reset(seed=goal_exposure_reset_seed(4, 1))
        self.assertTrue(all(torch.equal(left, right) for left, right in zip(before, agent.actor.parameters())))

    def test_exposure_reset_preserves_replay(self) -> None:
        item, _, _ = self._collector_transition()
        agent = GeneratorSAC(self.observation.size, GeneratorSACConfig(hidden_dim=16), seed=5)
        agent.observe(item)
        self.environment.reset(seed=goal_exposure_reset_seed(5, 1))
        self.assertEqual(len(agent.replay), 1)
        self.assertTrue(agent.replay.transitions[0].collector_boundary)

    def test_collector_reset_state_not_used_as_previous_successor(self) -> None:
        item, physical_successor, _ = self._collector_transition()
        reset_observation, _ = self.environment.reset(seed=goal_exposure_reset_seed(0, 1))
        np.testing.assert_array_equal(item.next_observation, physical_successor)
        self.assertFalse(np.array_equal(item.next_observation, reset_observation))

    def test_collector_boundary_preserves_real_successor(self) -> None:
        item, physical_successor, next_context = self._collector_transition()
        np.testing.assert_array_equal(item.next_observation, physical_successor)
        self.assertEqual(item.next_execution_authority, next_context["execution_authority"])
        self.assertEqual(item.next_certificate_epoch, next_context["certificate_epoch"])
        if item.next_generator_executable:
            np.testing.assert_array_equal(item.next_c, next_context["c"])
            np.testing.assert_array_equal(item.next_G, next_context["G"])

    def test_collector_boundary_does_not_force_terminal_target(self) -> None:
        item, _, _ = self._collector_transition()
        agent = PersistentGeneratorSAC(
            self.observation.size,
            GeneratorSACConfig(batch_size=2, hidden_dim=16, bootstrap_on_truncation=True),
            seed=6,
        )
        without_boundary = replace(item, collector_boundary=False)
        torch.manual_seed(17)
        boundary_targets, boundary_counts = agent.bellman_target([item, item])
        torch.manual_seed(17)
        ordinary_targets, ordinary_counts = agent.bellman_target([without_boundary, without_boundary])
        torch.testing.assert_close(boundary_targets, ordinary_targets)
        self.assertEqual(boundary_counts["generator_target_count"], ordinary_counts["generator_target_count"])
        self.assertEqual(boundary_counts["collector_boundary_target_count"], 2)

    def test_true_termination_still_zero_bootstraps(self) -> None:
        item, _, _ = self._collector_transition()
        terminated = replace(item, terminated=True, collector_boundary=True)
        agent = PersistentGeneratorSAC(
            self.observation.size,
            GeneratorSACConfig(batch_size=2, hidden_dim=16, bootstrap_on_truncation=True),
            seed=16,
        )
        targets, _ = agent.bellman_target([terminated, terminated])
        torch.testing.assert_close(targets, torch.full((2,), float(item.reward)))

    def test_truncation_semantics_unchanged(self) -> None:
        item, _, _ = self._collector_transition()
        item = replace(
            item,
            collector_boundary=False,
            truncated=True,
            next_generator_available=False,
            next_certificate_valid=False,
            next_c=None,
            next_G=None,
        )
        agent = GeneratorSAC(
            self.observation.size,
            GeneratorSACConfig(batch_size=2, hidden_dim=16, bootstrap_on_truncation=True),
            seed=7,
        )
        with torch.no_grad():
            for network in (agent.target_critic_1, agent.target_critic_2):
                for parameter in network.parameters():
                    parameter.zero_()
                network.network[-1].bias.fill_(1.0)
        targets, _ = agent.bellman_target([item, item])
        expected = float(item.reward) + agent.config.gamma
        torch.testing.assert_close(targets, torch.full((2,), expected))

        no_bootstrap_agent = GeneratorSAC(
            self.observation.size,
            GeneratorSACConfig(batch_size=2, hidden_dim=16, bootstrap_on_truncation=False),
            seed=7,
        )
        terminal_targets, _ = no_bootstrap_agent.bellman_target([item, item])
        torch.testing.assert_close(terminal_targets, torch.full((2,), float(item.reward)))

    def test_exposure_disabled_is_backward_compatible(self) -> None:
        self.assertEqual(training_protocol_name(None), "persistent_only")
        self.assertFalse(goal_exposure_reset_boundary(250, 5000, None, terminated=False, truncated=False))
        self.assertFalse(goal_exposure_reset_boundary(250, 5000, 0, terminated=False, truncated=False))

    def test_persistent_evaluation_never_uses_exposure_reset(self) -> None:
        goal_before = self.environment.task_env.manager.current_task.goal_position.copy()
        self.environment.task_env.manager.goal_radius = 1e-6
        self.environment.task_env.step(np.zeros(3, dtype=np.float64))
        np.testing.assert_array_equal(self.environment.task_env.manager.current_task.goal_position, goal_before)

    def test_collector_reset_is_not_natural_completion(self) -> None:
        tracker = GoalExposureAccumulator()
        tracker.assign(np.ones(3), np.zeros(3), 0, "initial_reset", 0)
        tracker.observe_step(False)
        tracker.assign(np.full(3, 2.0), np.zeros(3), 250, "collector_reset", 1_000_003)
        summary = tracker.summary()
        self.assertEqual(summary["collector_resets"], 1)
        self.assertEqual(summary["natural_task_completions"], 0)

    def test_batch_goal_diversity_uses_explicit_goal_metadata(self) -> None:
        item, _, _ = self._collector_transition()
        goal = np.array((1.2345678, 2.3456789, 1.0), dtype=np.float32)
        first = replace(item, task_goal=goal, observation=item.observation.copy())
        shifted = item.observation.copy()
        shifted[self.environment.task_env.observation_layout["position"]] += 1e-4
        second = replace(item, task_goal=goal, observation=shifted)
        metrics = batch_goal_diversity(
            [first, second],
            self.environment.task_env.observation_layout["position"],
            self.environment.task_env.observation_layout["goal_delta"],
            self.environment.plant.config.world_size,
        )
        self.assertEqual(metrics["batch_unique_goal_count"], 1)


if __name__ == "__main__":
    unittest.main()
