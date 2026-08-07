from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from cert_runtime.energy_management import (
    EnergyDecision,
    EnergyManagementReplayBuffer,
    EnergyManagementSAC,
    EnergyManagementTransition,
    ReserveOnlyEnergyPolicy,
)
from envs.certified_uav import (
    CertifiedGoalNetwork,
    CertifiedSingleUAVPlantEnv,
    ChargingConfig,
    ChargingDynamics,
    FixedCertificationScenario,
    GoalEdgeType,
    PersistentGoalCertificateProvider,
    PersistentGoalTaskManager,
    PersistentGoalWrapper,
    PersistentMissionMode,
    make_persistent_uav_env,
    verify_departure_energy,
)
from envs.certified_uav.config import CertifiedUAVConfig, apply_configuration_overrides


def persistent_fixture(name: str = "persistent_open.json"):
    scenario = FixedCertificationScenario(name).definition
    config = apply_configuration_overrides(
        CertifiedUAVConfig(world_size=scenario.world_size),
        scenario.configuration_overrides,
    )
    network = CertifiedGoalNetwork.from_config(scenario.mission_config["persistent"])
    plant = CertifiedSingleUAVPlantEnv(config, scenario)
    return scenario, config, network, plant


def complete_active_goal(manager: PersistentGoalTaskManager, step: int = 1):
    event = {"task_completed": False}
    while not event["task_completed"]:
        if manager.active_edge is None:
            raise AssertionError("pending goal has no active certified route")
        event = manager.advance(manager.network.nodes[manager.active_edge.target].position, step)
        step += 1
    return event, step


class ChargingDynamicsTests(unittest.TestCase):
    def test_docked_uav_charges_exactly_rate_times_dt_without_motion_reset(self):
        _, config, network, plant = persistent_fixture()
        plant.reset(seed=0)
        plant.state.position = network.nodes[network.charging_station].position.copy()
        plant.state.velocity = np.zeros(3)
        position = plant.state.position.copy()
        velocity = plant.state.velocity.copy()
        result = ChargingDynamics(ChargingConfig()).step(plant, "epoch")
        self.assertAlmostEqual(result.charged_energy, 2.0 * config.dt)
        np.testing.assert_array_equal(plant.state.position, position)
        np.testing.assert_array_equal(plant.state.velocity, velocity)

    def test_non_docked_or_moving_uav_cannot_charge(self):
        _, _, network, plant = persistent_fixture()
        plant.reset(seed=1)
        dynamics = ChargingDynamics()
        with self.assertRaisesRegex(RuntimeError, "CHARGING_NOT_ADMISSIBLE"):
            dynamics.step(plant, "epoch")
        plant.state.position = network.nodes[network.charging_station].position.copy()
        plant.state.velocity = np.array([0.06, 0.0, 0.0])
        with self.assertRaisesRegex(RuntimeError, "CHARGING_NOT_ADMISSIBLE"):
            dynamics.step(plant, "epoch")

    def test_charge_is_capacity_bounded_and_never_teleports(self):
        _, _, network, plant = persistent_fixture()
        plant.reset(seed=2)
        plant.state.position = network.nodes[network.charging_station].position.copy()
        plant.state.velocity = np.zeros(3)
        plant.state.energy = 29.9
        position = plant.state.position.copy()
        ChargingDynamics().step(plant, "epoch")
        self.assertEqual(plant.state.energy, 30.0)
        np.testing.assert_array_equal(plant.state.position, position)

    def test_station_arrival_does_not_terminate_persistent_but_can_terminate_single_mode(self):
        scenario, config, network, persistent_plant = persistent_fixture()
        persistent_plant.reset(seed=3)
        persistent_plant.state.position = network.nodes[network.charging_station].position.copy()
        persistent_plant.state.velocity = np.zeros(3)
        _, _, terminated, _, _ = persistent_plant.step(np.zeros(3))
        self.assertFalse(terminated)
        single_plant = CertifiedSingleUAVPlantEnv(replace(config, terminate_on_terminal=True), scenario)
        single_plant.reset(seed=3)
        single_plant.state.position = network.nodes[network.charging_station].position.copy()
        single_plant.state.velocity = np.zeros(3)
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


class PersistentGoalStreamTests(unittest.TestCase):
    def test_new_goal_differs_from_current_and_station_is_never_sampled(self):
        _, _, network, _ = persistent_fixture()
        manager = PersistentGoalTaskManager(network, 0.2, 10.0)
        manager.reset(0, network.nodes["A"].position)
        self.assertNotEqual(manager.current_task.goal_node, "A")
        self.assertNotEqual(manager.current_task.goal_node, network.charging_station)

    def test_goal_stream_is_seed_reproducible(self):
        _, _, network, _ = persistent_fixture()
        sequences = []
        for _ in range(2):
            manager = PersistentGoalTaskManager(network, 0.2, 10.0)
            manager.reset(7, network.nodes["A"].position)
            sequence = []
            step = 1
            for _ in range(5):
                sequence.append(manager.current_task.goal_node)
                _, step = complete_active_goal(manager, step)
            sequences.append(sequence)
        self.assertEqual(sequences[0], sequences[1])

    def test_multiple_tasks_complete_without_ending_stream(self):
        _, _, network, _ = persistent_fixture()
        manager = PersistentGoalTaskManager(network, 0.2, 10.0)
        manager.reset(1, network.nodes["A"].position)
        step = 1
        for _ in range(3):
            old_id = manager.current_task.task_id
            event, step = complete_active_goal(manager, step)
            self.assertTrue(event["task_completed"])
            self.assertNotEqual(manager.current_task.task_id, old_id)
        self.assertEqual(manager.tasks_completed, 3)

    def test_charging_interruption_preserves_pending_goal_and_departure_route(self):
        _, _, network, _ = persistent_fixture()
        manager = PersistentGoalTaskManager(network, 0.2, 10.0)
        manager.reset(2, network.nodes["A"].position)
        task_id = manager.current_task.task_id
        goal_id = manager.current_task.goal_node
        manager.interrupt_for_charge()
        manager.mark_station_arrival()
        manager.resume_from_station()
        self.assertEqual(manager.current_task.task_id, task_id)
        self.assertEqual(manager.current_task.goal_node, goal_id)
        self.assertTrue(manager.current_task.interrupted_by_charge)
        self.assertEqual(manager.active_edge.edge_type, GoalEdgeType.DEPARTURE_EDGE)

    def test_only_pending_goal_completes_task(self):
        _, _, network, _ = persistent_fixture()
        manager = PersistentGoalTaskManager(network, 0.2, 10.0)
        manager.reset(3, network.nodes["A"].position)
        event = manager.advance(network.nodes[network.charging_station].position, 1)
        self.assertFalse(event["task_completed"])
        self.assertEqual(manager.tasks_completed, 0)

    def test_forced_and_voluntary_return_do_not_complete_or_replace_task(self):
        _, _, network, plant = persistent_fixture()
        wrapper = PersistentGoalWrapper(plant, network)
        wrapper.reset(seed=4)
        task_id = wrapper.manager.current_task.task_id
        wrapper.request_return(forced=False)
        self.assertEqual(wrapper.mode, PersistentMissionMode.VOLUNTARY_RETURN)
        self.assertEqual(wrapper.manager.current_task.task_id, task_id)
        self.assertEqual(wrapper.manager.tasks_completed, 0)
        wrapper.mode = PersistentMissionMode.TASK
        wrapper.request_return(forced=True)
        self.assertEqual(wrapper.mode, PersistentMissionMode.FORCED_RETURN)
        self.assertEqual(wrapper.manager.current_task.task_id, task_id)
        self.assertEqual(wrapper.manager.tasks_completed, 0)

    def test_energy_management_decision_cannot_select_or_replace_goal(self):
        _, _, network, _ = persistent_fixture()
        manager = PersistentGoalTaskManager(network, 0.2, 10.0)
        manager.reset(5, network.nodes["A"].position)
        task_id = manager.current_task.task_id
        goal_id = manager.current_task.goal_node
        policy = ReserveOnlyEnergyPolicy()
        decision = policy.select_action(np.zeros(13), {"charging": False, "departure_allowed": True})
        self.assertEqual(decision, EnergyDecision.SERVE_OR_LEAVE)
        self.assertEqual(manager.current_task.task_id, task_id)
        self.assertEqual(manager.current_task.goal_node, goal_id)

    def test_task_recovery_and_departure_routes_are_typed_independently(self):
        _, _, network, _ = persistent_fixture()
        self.assertTrue(network.edges_of_type(GoalEdgeType.TASK_EDGE))
        self.assertTrue(network.edges_of_type(GoalEdgeType.RECOVERY_EDGE))
        self.assertTrue(network.edges_of_type(GoalEdgeType.DEPARTURE_EDGE))
        for edge in network.edges_of_type(GoalEdgeType.TASK_EDGE):
            self.assertNotIn(network.charging_station, (edge.source, edge.target))
        for goal in network.goal_node_ids:
            self.assertTrue(network.shortest_path(goal, network.charging_station, {GoalEdgeType.RECOVERY_EDGE}))
            self.assertTrue(network.shortest_path(network.charging_station, goal, {GoalEdgeType.DEPARTURE_EDGE}))

    def test_departure_gate_uses_pending_goal_departure_and_return_routes(self):
        scenario, _, network, _ = persistent_fixture()
        manager = PersistentGoalTaskManager(network, 0.2, 10.0)
        manager.reset(6, network.nodes["A"].position)
        provider = object.__new__(PersistentGoalCertificateProvider)
        provider.network = network
        provider.runtime = SimpleNamespace(scenario=scenario)
        provider.edge_energy_upper = {edge.edge_id: edge.energy_upper for edge in network.edges.values()}
        required = provider.required_departure_energy(manager.current_task)
        expected = (
            network.path_energy_upper(network.charging_station, manager.current_task.goal_node, {GoalEdgeType.DEPARTURE_EDGE})
            + network.path_energy_upper(manager.current_task.goal_node, network.charging_station, {GoalEdgeType.RECOVERY_EDGE})
            + scenario.terminal.minimum_energy
        )
        self.assertAlmostEqual(required, expected)


class EnergyManagementSMDPTests(unittest.TestCase):
    def transition(self, manifest: str = "manifest", duration: int = 7) -> EnergyManagementTransition:
        return EnergyManagementTransition(
            np.zeros(13, dtype=np.float32),
            1,
            0,
            3.0,
            duration,
            np.ones(13, dtype=np.float32),
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
        self.assertGreater(transition.duration_steps, 0)

    def test_bellman_discount_uses_gamma_power_duration(self):
        duration = torch.tensor([1.0, 5.0])
        actual = EnergyManagementSAC.smdp_discount(0.99, duration)
        expected = torch.tensor([0.99, 0.99 ** 5])
        self.assertTrue(torch.allclose(actual, expected))

    def test_energy_management_actor_is_not_part_of_generator_density(self):
        policy = EnergyManagementSAC(13)
        self.assertFalse(policy.uses_generator_density)

    def test_replay_rejects_wrong_manifest_and_does_not_alias(self):
        replay = EnergyManagementReplayBuffer(4, {"persistent_open": "manifest"})
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
    def test_all_persistent_scenarios_use_four_goals_and_station_is_separate(self):
        for name in ("persistent_open.json", "persistent_obstacle.json", "persistent_energy_tight.json"):
            scenario = FixedCertificationScenario(name).definition
            config = apply_configuration_overrides(
                CertifiedUAVConfig(world_size=scenario.world_size),
                scenario.configuration_overrides,
            )
            network = CertifiedGoalNetwork.from_config(scenario.mission_config["persistent"])
            self.assertEqual(len(network.goal_node_ids), 4)
            self.assertNotIn(network.charging_station, network.goal_node_ids)
            self.assertFalse(config.terminate_on_terminal)
            self.assertEqual(config.episode_limit, 5000)
            self.assertFalse(scenario.consistency_failures(config))

    def test_factory_does_not_eagerly_build_persistent_manifest(self):
        environment = make_persistent_uav_env("persistent_open.json")
        self.assertIsNone(environment.runtime.mission_provider)
        self.assertIsNone(environment.certificate_provider)

    def test_single_mission_default_terminal_semantics_are_unchanged(self):
        scenario = FixedCertificationScenario("mission_open.json").definition
        config = apply_configuration_overrides(
            CertifiedUAVConfig(world_size=scenario.world_size),
            scenario.configuration_overrides,
        )
        self.assertTrue(config.terminate_on_terminal)


if __name__ == "__main__":
    unittest.main()
