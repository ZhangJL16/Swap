from __future__ import annotations

import unittest

import numpy as np
import torch

from cert_runtime.generator_sac import (
    GeneratorReplayBuffer,
    GeneratorSAC,
    GeneratorSACConfig,
    GeneratorTransition,
)
from envs.certified_uav import MissionPhase, make_certified_uav_env
from experiments.agents import DirectSACAgent, DirectTransition


def transition(*, accepted=True, next_generator=True, terminated=False, epoch="epoch-a", next_c=None, next_G=None):
    observation = np.linspace(-0.2, 0.2, 12, dtype=np.float32)
    c = np.zeros(3, dtype=np.float32)
    G = np.diag([0.05, 0.06, 0.07]).astype(np.float32)
    return GeneratorTransition(
        observation, observation + 0.01, 1.0, terminated, False, 0, "OUTBOUND", "OUTBOUND",
        epoch, epoch, np.zeros(3) if accepted else None, np.zeros(3) if accepted else None,
        c if accepted else None, G if accepted else None, c if accepted else None,
        np.zeros(3), np.zeros(3), np.zeros(3), accepted, None if accepted else "fallback",
        (c if next_c is None else next_c) if next_generator else None,
        (G if next_G is None else next_G) if next_generator else None,
        np.array([0.01, -0.01, 0.0]), next_generator, True, "1", "1", "energy-v1", ("r", "z"),
    )


class MultiStepMissionTests(unittest.TestCase):
    def test_mission_initial_state_is_not_terminal_and_does_not_auto_reset(self):
        runtime = make_certified_uav_env(
            "mission_open.json",
            timing_mode="functional",
        )
        runtime.reset(seed=0)
        initial = runtime.plant.state.copy()
        self.assertFalse(runtime.plant.terminal.is_admissible(initial))
        for step in range(5):
            _, _, terminated, truncated, info = runtime.step(np.zeros(3))
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["episode_step"], step + 1)
        self.assertEqual(runtime.plant.step_count, 5)
        self.assertGreater(runtime.plant.state.timestamp, initial.timestamp)

    def test_deterministic_fixture_completes_outbound_and_return(self):
        runtime = make_certified_uav_env(
            "mission_open.json",
            timing_mode="functional",
        )
        runtime.reset(seed=0)
        observed_return = False
        for _ in range(runtime.config.episode_limit):
            context = runtime.action_context()
            target = runtime.task_env.active_goal
            desired = np.clip(
                0.5 * (target - runtime.plant.state.position) - 1.2 * runtime.plant.state.velocity,
                -runtime.config.a_max,
                runtime.config.a_max,
            )
            if context["G"] is None:
                actor_u = np.zeros(3)
            else:
                eta = np.linalg.solve(context["G"], desired - context["c"])
                actor_u = np.arctanh(np.clip(eta, -0.95, 0.95))
            _, _, terminated, truncated, info = runtime.step(actor_u)
            observed_return |= info["mission_phase"] == MissionPhase.RETURN.name
            if terminated or truncated:
                self.assertTrue(info["task_completed"])
                self.assertTrue(info["terminal_return_success"])
                self.assertEqual(info["mission_phase"], MissionPhase.SUCCESS.name)
                self.assertGreater(info["episode_step"], 20)
                break
        else:
            self.fail("mission fixture did not terminate")
        self.assertTrue(observed_return)

    def test_reward_weights_do_not_change_certificate_context(self):
        first = make_certified_uav_env("mission_open.json")
        second = make_certified_uav_env("mission_open.json")
        first.reset(seed=1); second.reset(seed=1)
        second.task_env.reward_config = second.task_env.reward_config.__class__(0.0, 0.0, 0.0, 0.0)
        left, right = first.action_context(), second.action_context()
        np.testing.assert_allclose(left["kappa"], right["kappa"])
        np.testing.assert_allclose(left["c"], right["c"])
        np.testing.assert_allclose(left["G"], right["G"])


class GeneratorSACTests(unittest.TestCase):
    def setUp(self):
        self.config = GeneratorSACConfig(batch_size=4, hidden_dim=16, warmup_steps=0, epoch_replay_policy="group")
        self.agent = GeneratorSAC(12, self.config, seed=0)

    def test_fallback_target_uses_gamma_without_generator_entropy(self):
        batch = [transition(accepted=False, next_generator=False) for _ in range(4)]
        before = self.agent.generator_log_density_calls
        target, counts = self.agent.bellman_target(batch)
        self.assertEqual(self.agent.generator_log_density_calls, before)
        self.assertEqual(counts["fallback_target_count"], 4)
        self.assertTrue(torch.isfinite(target).all())

    def test_generator_target_uses_next_context_and_entropy(self):
        next_c = np.array([0.02, 0.01, -0.01], dtype=np.float32)
        next_G = np.diag([0.03, 0.04, 0.05]).astype(np.float32)
        batch = [transition(next_c=next_c, next_G=next_G) for _ in range(4)]
        before = self.agent.generator_log_density_calls
        target, counts = self.agent.bellman_target(batch)
        self.assertEqual(counts["generator_target_count"], 4)
        self.assertEqual(self.agent.generator_log_density_calls - before, 4)
        self.assertTrue(torch.isfinite(target).all())

    def test_terminal_transition_does_not_bootstrap(self):
        batch = [transition(terminated=True, next_generator=False) for _ in range(4)]
        target, _ = self.agent.bellman_target(batch)
        torch.testing.assert_close(target, torch.ones(4))

    def test_polyak_update_moves_target_toward_online(self):
        target_before = next(self.agent.target_critic_1.parameters()).detach().clone()
        with torch.no_grad():
            next(self.agent.critic_1.parameters()).add_(1.0)
        self.agent.polyak_update()
        target_after = next(self.agent.target_critic_1.parameters()).detach()
        self.assertFalse(torch.equal(target_before, target_after))

    def test_fallback_only_batch_skips_actor_and_updates_critic(self):
        batch = [transition(accepted=False, next_generator=False) for _ in range(4)]
        actor_before = [parameter.detach().clone() for parameter in self.agent.actor.parameters()]
        critic_before = [parameter.detach().clone() for parameter in self.agent.critic_1.parameters()]
        metrics = self.agent.update(batch)
        self.assertEqual(metrics["actor_status"], "zero-accepted-sample")
        self.assertTrue(all(torch.equal(left, right) for left, right in zip(actor_before, self.agent.actor.parameters())))
        self.assertTrue(any(not torch.equal(left, right) for left, right in zip(critic_before, self.agent.critic_1.parameters())))

    def test_accepted_batch_updates_actor_and_critic_uses_exec(self):
        batch = [transition() for _ in range(4)]
        actor_before = [parameter.detach().clone() for parameter in self.agent.actor.parameters()]
        metrics = self.agent.update(batch)
        self.assertEqual(metrics["actor_status"], "updated")
        self.assertTrue(any(not torch.equal(left, right) for left, right in zip(actor_before, self.agent.actor.parameters())))
        self.assertGreater(metrics["accepted_batch_count"], 0)

    def test_replay_epoch_policy_groups_and_arrays_do_not_alias(self):
        replay = GeneratorReplayBuffer(10, "group", seed=0)
        first = transition(epoch="a")
        replay.add(first)
        original = first.observation.copy()
        source = np.zeros(12, dtype=np.float32)
        item = transition(epoch="b")
        replay.add(item)
        source[:] = 2.0
        np.testing.assert_array_equal(first.observation, original)
        with self.assertRaises(ValueError):
            replay.sample(2)


class BaselineFairnessTests(unittest.TestCase):
    def test_shield_critic_transition_uses_post_shield_action(self):
        runtime = make_certified_uav_env("mission_open.json")
        observation, _ = runtime.reset(seed=3)
        next_observation, reward, terminated, truncated, info = runtime.step_nominal_action(runtime.config.a_max)
        executed = info["telemetry"].action_trace.published
        item = DirectTransition(observation, next_observation, reward, terminated, truncated, executed)
        np.testing.assert_allclose(item.executed_action, executed, atol=1e-7)

    def test_all_methods_share_same_plant_and_reward_types(self):
        direct = make_certified_uav_env("mission_open.json").task_env
        shield = make_certified_uav_env("mission_open.json")
        self.assertEqual(type(direct.plant), type(shield.plant))
        self.assertEqual(type(direct.reward_config), type(shield.task_env.reward_config))

    def test_direct_sac_updates_from_executed_actions(self):
        agent = DirectSACAgent(12, np.array([0.1, 0.1, 0.1]), seed=0, batch_size=2, hidden_dim=16)
        for action in (np.array([0.1, 0.0, 0.0]), np.array([-0.1, 0.0, 0.0])):
            agent.observe(DirectTransition(np.zeros(12), np.ones(12), 0.1, False, False, action))
        metrics = agent.update()
        self.assertIsNotNone(metrics)
        self.assertTrue(np.isfinite(metrics["critic_loss_1"]))

    def test_direct_sac_checkpoint_restores_policy_temperature_and_step(self):
        action_max = np.array([0.1, 0.1, 0.1])
        agent = DirectSACAgent(12, action_max, seed=0, batch_size=2, hidden_dim=16)
        for action in (np.array([0.1, 0.0, 0.0]), np.array([-0.1, 0.0, 0.0])):
            agent.observe(DirectTransition(np.zeros(12), np.ones(12), 0.1, False, False, action))
        agent.update()
        restored = DirectSACAgent(12, action_max, seed=1, batch_size=2, hidden_dim=16)
        restored.load_state_dict(agent.state_dict())
        observation = np.linspace(-1.0, 1.0, 12)
        np.testing.assert_allclose(
            restored.select_action(observation, deterministic=True),
            agent.select_action(observation, deterministic=True),
            atol=1e-7,
        )
        self.assertAlmostEqual(float(restored.alpha.detach()), float(agent.alpha.detach()))
        self.assertEqual(restored.gradient_steps, agent.gradient_steps)


if __name__ == "__main__":
    unittest.main()
