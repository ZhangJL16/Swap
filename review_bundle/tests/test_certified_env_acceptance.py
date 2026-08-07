from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from cert_runtime.smoke_training import MinimalGeneratorSAC, SmokeTransition, density_gradient_acceptance
from envs.certified_uav import make_certified_uav_env
from envs.certified_uav.acceptance import SCENARIO_EXPECTATIONS, run_acceptance_cycle
from envs.certified_uav.obstacles import AABBObstacle
from envs.certified_uav.task_wrapper import TaskRewardConfig


class AcceptanceTraceTests(unittest.TestCase):
    def test_open_trace_closes_execution_invariants(self):
        trace, _ = run_acceptance_cycle("open_corridor", 0)
        self.assertTrue(trace["accepted"])
        self.assertTrue(trace["actor_called"])
        self.assertTrue(trace["plant_input_matches_exec"])
        self.assertTrue(trace["published_once"])
        self.assertTrue(trace["manifest_integrity"])

    def test_failure_matrix_fails_closed(self):
        for scenario, reason in SCENARIO_EXPECTATIONS.items():
            if scenario == "open_corridor":
                continue
            with self.subTest(scenario=scenario):
                trace, _ = run_acceptance_cycle(scenario, 1)
                self.assertFalse(trace["accepted"])
                self.assertEqual(trace["fallback_reason"], reason)
                np.testing.assert_array_equal(trace["a_exec"], trace["kappa"])

    def test_reward_parameters_and_goal_do_not_change_certificate_source(self):
        first = make_certified_uav_env()
        second = make_certified_uav_env()
        first.reset(seed=3)
        second.reset(seed=3)
        second.task_env.reward_config = TaskRewardConfig(0.0, 0.0, 0.0, 0.0)
        second.task_env.goal = np.array([0.1, 3.9, 1.5])
        first.step(np.array([0.1, -0.2, 0.3]))
        second.step(np.array([0.1, -0.2, 0.3]))
        left, right = first.replay.records[-1], second.replay.records[-1]
        self.assertEqual(left.recovery_action, right.recovery_action)
        self.assertEqual(left.zonotope_center, right.zonotope_center)
        self.assertEqual(left.zonotope_generators, right.zonotope_generators)
        self.assertEqual(left.accepted, right.accepted)
        self.assertEqual(left.fallback_reason, right.fallback_reason)

    def test_world_change_does_not_magically_update_certificate_grid(self):
        runtime = make_certified_uav_env()
        runtime.reset(seed=4)
        before = runtime.geometry.certificate_digest()
        runtime.plant.world.aabbs = runtime.plant.world.aabbs + (
            AABBObstacle(np.array([1.0, 0.73, 0.0]), np.array([1.04, 0.77, 2.0])),
        )
        self.assertEqual(before, runtime.geometry.certificate_digest())
        packet = runtime.plant.lidar_model.measure(runtime.plant.state, runtime.plant.world, runtime.plant.np_random)
        runtime.geometry.update_lidar(
            tuple(packet.pose_position[:2]), packet.to_certificate_rays(), runtime.sensor_bounds, packet.timestamp
        )
        self.assertNotEqual(before, runtime.geometry.certificate_digest())


class TrainingSemanticsTests(unittest.TestCase):
    def _accepted_transition(self):
        runtime = make_certified_uav_env(
            freeze_certificate_epoch=True,
            timing_mode="functional",
        )
        observation, _ = runtime.reset(seed=5)
        next_observation, reward, terminated, truncated, _ = runtime.step(np.zeros(3))
        record = runtime.replay.records[-1]
        return runtime, SmokeTransition(observation, next_observation, reward, terminated or truncated, record)

    def test_minimal_trainer_updates_actor_and_critics(self):
        runtime, transition = self._accepted_transition()
        self.assertTrue(
            transition.record.accepted,
            msg=(
                f"fallback_reason={transition.record.fallback_reason}, "
                f"timing_mode={runtime.timing_mode}, "
                f"watchdog_trace={runtime.watchdog.last_trace}, "
                f"stage_timings={runtime.last_stage_timings}"
            ),
        )
        trainer = MinimalGeneratorSAC(transition.observation.size, 0)
        trainer.freeze_epoch(transition.record)
        before_actor = [parameter.detach().clone() for parameter in trainer.actor.parameters()]
        before_critic = [parameter.detach().clone() for parameter in trainer.critic_1.parameters()]
        result = trainer.update((transition,))
        self.assertEqual(
            result["actor_status"],
            "updated",
            msg=(
                f"accepted={transition.record.accepted}, "
                f"fallback_reason={transition.record.fallback_reason}, "
                f"timing_mode={runtime.timing_mode}"
            ),
        )
        self.assertTrue(any(not torch_equal(a, b) for a, b in zip(before_actor, trainer.actor.parameters())))
        self.assertTrue(any(not torch_equal(a, b) for a, b in zip(before_critic, trainer.critic_1.parameters())))

    def test_fallback_only_batch_never_calls_generator_density(self):
        runtime, transition = self._accepted_transition()
        fallback_record = replace(
            transition.record,
            accepted=False,
            nominal_pre_squash_u=None,
            squashed_eta=None,
            candidate_action=None,
            zonotope_center=None,
            zonotope_generators=None,
            inclusion_certificate_hash=None,
            fallback_reason="TEST_FALLBACK",
        )
        trainer = MinimalGeneratorSAC(transition.observation.size, 1)
        trainer.freeze_epoch(fallback_record)
        critic_before = [parameter.detach().clone() for parameter in trainer.critic_1.parameters()]
        result = trainer.update((replace(transition, record=fallback_record),))
        self.assertEqual(result["actor_status"], "zero-accepted-sample")
        self.assertEqual(trainer.generator_log_density_calls, 0)
        self.assertTrue(any(not torch_equal(a, b) for a, b in zip(critic_before, trainer.critic_1.parameters())))
        self.assertEqual(runtime.replay.records[-1].critic_action, runtime.replay.records[-1].executed_action)

    def test_mixed_epoch_is_rejected(self):
        _, transition = self._accepted_transition()
        trainer = MinimalGeneratorSAC(transition.observation.size, 2)
        trainer.freeze_epoch(transition.record)
        stale_snapshot = replace(transition.record.certificate_state, local_geometry_digest="stale")
        stale_record = replace(transition.record, certificate_state=stale_snapshot)
        with self.assertRaises(ValueError):
            trainer.update((replace(transition, record=stale_record),))

    def test_density_formula_and_gradient_acceptance(self):
        result = density_gradient_acceptance(3)
        self.assertLess(result["maximum_formula_absolute_error"], 1e-9)
        self.assertLess(result["maximum_gradient_absolute_error"], 1e-4)


def torch_equal(left, right) -> bool:
    import torch

    return torch.equal(left, right.detach())


if __name__ == "__main__":
    unittest.main()
