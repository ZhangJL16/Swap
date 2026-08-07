from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
import torch

from cert_runtime.charging_scheduler import (
    ChargingSchedulerSAC,
    SchedulerReplayBuffer,
    SchedulerTransition,
)
from envs.certified_uav import (
    CertifiedServiceNetwork,
    CertifiedSingleUAVPlantEnv,
    ChargingConfig,
    ChargingDynamics,
    FixedCertificationScenario,
    PersistentMissionMode,
    PersistentTaskManager,
    PersistentTaskStatus,
    PersistentTaskWrapper,
    make_persistent_uav_env,
    verify_departure_energy,
)
from envs.certified_uav.config import CertifiedUAVConfig, apply_configuration_overrides


def persistent_fixture():
    scenario = FixedCertificationScenario("persistent_open.json").definition
    config = apply_configuration_overrides(CertifiedUAVConfig(world_size=scenario.world_size), scenario.configuration_overrides)
    network = CertifiedServiceNetwork.from_config(scenario.mission_config["persistent"])
    plant = CertifiedSingleUAVPlantEnv(config, scenario)
    return scenario, config, network, plant


class ChargingDynamicsTests(unittest.TestCase):
    def test_docked_uav_charges_exactly_rate_times_dt_without_motion_reset(self):
        _, config, _, plant = persistent_fixture()
        plant.reset(seed=0)
        position = plant.state.position.copy()
        velocity = plant.state.velocity.copy()
        result = ChargingDynamics(ChargingConfig()).step(plant, "epoch")
        self.assertAlmostEqual(result.charged_energy, 2.0 * config.dt)
        np.testing.assert_array_equal(plant.state.position, position)
        np.testing.assert_array_equal(plant.state.velocity, velocity)

    def test_non_docked_or_moving_uav_cannot_charge(self):
        _, _, _, plant = persistent_fixture()
        plant.reset(seed=1)
        dynamics = ChargingDynamics()
        plant.state.position = np.array([2.0, 2.0, 1.0])
        with self.assertRaisesRegex(RuntimeError, "CHARGING_NOT_ADMISSIBLE"):
            dynamics.step(plant, "epoch")
        plant.state.position = plant.scenario.station_position.copy()
        plant.state.velocity = np.array([0.06, 0.0, 0.0])
        with self.assertRaisesRegex(RuntimeError, "CHARGING_NOT_ADMISSIBLE"):
            dynamics.step(plant, "epoch")

    def test_charge_is_capacity_bounded_and_never_teleports(self):
        _, _, _, plant = persistent_fixture()
        plant.reset(seed=2)
        plant.state.energy = 29.9
        position = plant.state.position.copy()
        ChargingDynamics().step(plant, "epoch")
        self.assertEqual(plant.state.energy, 30.0)
        np.testing.assert_array_equal(plant.state.position, position)

    def test_station_arrival_does_not_terminate_persistent_but_can_terminate_single_mode(self):
        scenario, config, _, persistent_plant = persistent_fixture()
        persistent_plant.reset(seed=3)
        _, _, terminated, _, _ = persistent_plant.step(np.zeros(3))
        self.assertFalse(terminated)
        single_plant = CertifiedSingleUAVPlantEnv(replace(config, terminate_on_terminal=True), scenario)
        single_plant.reset(seed=3)
        _, _, terminated, _, _ = single_plant.step(np.zeros(3))
        self.assertTrue(terminated)

    def test_departure_energy_gate_rejects_and_accepts_without_reward_logic(self):
        rejected = verify_departure_energy(4.9, 4.5, 0.5, True)
        accepted = verify_departure_energy(5.0, 4.5, 0.5, True)
        invalid = verify_departure_energy(30.0, 4.5, 0.5, False)
        self.assertFalse(rejected.allowed)
        self.assertEqual(rejected.reason, "INSUFFICIENT_DEPARTURE_ENERGY")
        self.assertTrue(accepted.allowed)
        self.assertEqual(invalid.reason, "PERSISTENT_CERTIFICATE_INVALID")


class PersistentTaskTests(unittest.TestCase):
    def test_task_survives_charge_pause_and_resume(self):
        _, _, network, _ = persistent_fixture()
        manager = PersistentTaskManager(network, 0.2, 10.0)
        manager.reset(0, network.nodes[network.charging_station].position)
        task_id = manager.current_task.task_id
        prior_status = manager.current_task.status
        manager.pause_for_charge()
        self.assertEqual(manager.current_task.status, PersistentTaskStatus.PAUSED_FOR_CHARGE)
        manager.resume_from_station()
        self.assertEqual(manager.current_task.task_id, task_id)
        self.assertEqual(manager.current_task.status, prior_status)
        self.assertEqual(manager.task_pause_count, 1)
        self.assertEqual(manager.task_resume_count, 1)

    def test_completion_assigns_next_task_and_does_not_end_stream(self):
        _, _, network, _ = persistent_fixture()
        manager = PersistentTaskManager(network, 0.2, 10.0)
        manager.reset(0, network.nodes[network.charging_station].position)
        first_id = manager.current_task.task_id
        manager.advance(network.nodes["remote"].position)
        self.assertEqual(manager.tasks_completed, 1)
        self.assertNotEqual(manager.current_task.task_id, first_id)
        manager.advance(network.nodes["station"].position)
        self.assertEqual(manager.tasks_completed, 2)
        self.assertIsNotNone(manager.current_task)

    def test_forced_return_overrides_service_mode_without_deleting_task(self):
        _, _, network, plant = persistent_fixture()
        wrapper = PersistentTaskWrapper(plant, network)
        wrapper.reset(seed=4)
        task_id = wrapper.manager.current_task.task_id
        wrapper.on_runtime_recovery("ENERGY_MARGIN_FORCED_RETURN")
        self.assertEqual(wrapper.mode, PersistentMissionMode.FORCED_RETURN)
        self.assertEqual(wrapper.manager.current_task.task_id, task_id)
        self.assertEqual(wrapper.manager.current_task.status, PersistentTaskStatus.PAUSED_FOR_CHARGE)

    def test_scheduler_decision_points_are_event_based(self):
        _, _, network, _ = persistent_fixture()
        manager = PersistentTaskManager(network, 0.2, 10.0)
        manager.reset(5, network.nodes[network.charging_station].position)
        self.assertTrue(manager.decision_required)
        manager.accept_service_decision()
        self.assertFalse(manager.decision_required)
        manager.advance(network.nodes["remote"].position)
        self.assertTrue(manager.decision_required)


class SchedulerSMDPTests(unittest.TestCase):
    def transition(self, manifest: str = "manifest", duration: int = 7) -> SchedulerTransition:
        return SchedulerTransition(
            np.zeros(14, dtype=np.float32),
            1,
            0,
            3.0,
            duration,
            np.ones(14, dtype=np.float32),
            False,
            True,
            "ENERGY_MARGIN_FORCED_RETURN",
            "persistent_open",
            manifest,
        )

    def test_requested_and_executed_modes_and_override_are_distinct(self):
        transition = self.transition()
        self.assertEqual(transition.requested_decision, 1)
        self.assertEqual(transition.executed_decision, 0)
        self.assertTrue(transition.forced_override)
        self.assertIsNotNone(transition.override_reason)
        self.assertGreater(transition.duration_steps, 0)

    def test_bellman_discount_uses_gamma_power_duration(self):
        duration = torch.tensor([1.0, 5.0])
        actual = ChargingSchedulerSAC.smdp_discount(0.99, duration)
        expected = torch.tensor([0.99, 0.99 ** 5])
        self.assertTrue(torch.allclose(actual, expected))

    def test_scheduler_is_not_part_of_generator_density(self):
        scheduler = ChargingSchedulerSAC(14)
        self.assertFalse(scheduler.uses_generator_density)

    def test_scheduler_replay_rejects_wrong_manifest_and_does_not_alias(self):
        replay = SchedulerReplayBuffer(4, {"persistent_open": "manifest"})
        with self.assertRaisesRegex(ValueError, "scenario/manifest mismatch"):
            replay.add(self.transition("wrong"))
        transition = self.transition()
        replay.add(transition)
        original = transition.observation.copy()
        transition.observation[0] = 9.0
        np.testing.assert_array_equal(replay.records[0].observation, original)

    def test_zero_duration_transition_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duration"):
            self.transition(duration=0)


class PersistentScenarioTests(unittest.TestCase):
    def test_all_persistent_scenarios_are_separate_and_nonterminal_at_station(self):
        for name in ("persistent_open.json", "persistent_obstacle.json", "persistent_energy_tight.json"):
            scenario = FixedCertificationScenario(name).definition
            config = apply_configuration_overrides(CertifiedUAVConfig(world_size=scenario.world_size), scenario.configuration_overrides)
            self.assertTrue(scenario.mission_config["persistent"])
            self.assertFalse(config.terminate_on_terminal)
            self.assertEqual(config.episode_limit, 5000)

    def test_factory_does_not_eagerly_build_single_mission_provider(self):
        environment = make_persistent_uav_env("persistent_open.json")
        self.assertIsNone(environment.runtime.mission_provider)
        self.assertIsNone(environment.certificate_provider)


if __name__ == "__main__":
    unittest.main()
