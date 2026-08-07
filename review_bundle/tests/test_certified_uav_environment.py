from __future__ import annotations

from dataclasses import replace
import inspect
import unittest

import numpy as np

from cert_runtime import SimulatedWatchdog, WCETContract
from envs.certified_uav import (
    ActuatorTrackingModel,
    CertifiedSingleUAVPlantEnv,
    CertifiedTaskWrapper,
    CertifiedUAVConfig,
    EnergyModel,
    FixedCertificationScenario,
    HorizontalLidarModel,
    TerminalSpec,
    UAVPhysicalState,
    integrate_double_integrator,
    make_certified_uav_env,
)
from envs.certified_uav.config import apply_configuration_overrides
from envs.certified_uav.obstacles import AABBObstacle, StaticWorld


def build_plant(scenario_name: str = "open_corridor.json", **config_changes):
    scenario = FixedCertificationScenario(scenario_name).definition
    base = CertifiedUAVConfig(world_size=scenario.world_size, **config_changes)
    config = apply_configuration_overrides(base, scenario.configuration_overrides)
    return CertifiedSingleUAVPlantEnv(config, scenario), scenario, config


class UnifiedDynamicsTests(unittest.TestCase):
    def test_zero_action_preserves_velocity(self):
        position, velocity = integrate_double_integrator(
            np.array([1.0, 2.0, 3.0]),
            np.array([0.2, -0.1, 0.05]),
            np.zeros(3),
            0.2,
        )
        np.testing.assert_allclose(position, [1.04, 1.98, 3.01])
        np.testing.assert_allclose(velocity, [0.2, -0.1, 0.05])

    def test_constant_acceleration_formula(self):
        position, velocity = integrate_double_integrator(np.zeros(3), np.zeros(3), np.ones(3), 0.2)
        np.testing.assert_allclose(position, np.full(3, 0.02))
        np.testing.assert_allclose(velocity, np.full(3, 0.2))

    def test_calibration_residual_uses_authoritative_integrator(self):
        from calibration.dynamics import DynamicsSample
        from calibration.schema import DataSplit

        predicted_position, predicted_velocity = integrate_double_integrator(
            np.zeros(3), np.array([0.1, 0.0, 0.0]), np.array([0.02, 0.0, 0.0]), 0.2
        )
        sample = DynamicsSample(
            "shared-formula", 0.0, 0.2, (0.0, 0.0, 0.0), (0.1, 0.0, 0.0),
            tuple(predicted_position), tuple(predicted_velocity), (0.02, 0.0, 0.0),
            (0.02, 0.0, 0.0), (0.02, 0.0, 0.0), DataSplit.VALIDATION,
            (("speed", 0.1),),
        )
        self.assertEqual(sample.position_residual, (0.0, 0.0, 0.0))
        self.assertEqual(sample.velocity_residual, (0.0, 0.0, 0.0))


class PlantActionAndCollisionTests(unittest.TestCase):
    def test_action_box_rejects_out_of_bounds_without_clipping(self):
        plant, _, config = build_plant()
        plant.reset(seed=1)
        with self.assertRaises(ValueError):
            plant.step(config.a_max + np.array([1e-3, 0.0, 0.0]))

    def test_box_corner_is_not_rescaled_to_a_sphere(self):
        plant, _, config = build_plant()
        plant.reset(seed=1)
        corner = config.a_max.copy()
        plant.step(corner)
        np.testing.assert_allclose(plant.last_telemetry.action_trace.published, corner)
        self.assertGreater(np.linalg.norm(corner), float(np.max(config.a_max)))

    def test_measured_action_drives_the_plant(self):
        plant, scenario, config = build_plant()
        bias = np.array([0.001, -0.001, 0.0005])
        plant = CertifiedSingleUAVPlantEnv(
            config,
            scenario,
            actuator_model=ActuatorTrackingModel(config.tracking_error_bound, bias),
        )
        plant.reset(seed=2)
        before = plant.state.copy()
        command = np.zeros(3)
        plant.step(command)
        expected_position, expected_velocity = integrate_double_integrator(
            before.position, before.velocity, bias, config.dt
        )
        np.testing.assert_allclose(plant.state.position, expected_position)
        np.testing.assert_allclose(plant.state.velocity, expected_velocity)

    def test_swept_collision_detects_thin_crossing(self):
        world = StaticWorld(
            np.array([4.0, 4.0, 2.0]),
            (AABBObstacle(np.array([1.03, 0.7, 0.8]), np.array([1.031, 0.8, 1.2])),),
        )
        self.assertTrue(world.swept_collision(np.array([1.02, 0.75, 1.0]), np.array([1.044, 0.75, 1.0]), 0.0))
        self.assertFalse(np.all(np.array([1.044, 0.75, 1.0]) >= np.array([1.03, 0.7, 0.8])) and np.all(np.array([1.044, 0.75, 1.0]) <= np.array([1.031, 0.8, 1.2])))

    def test_collision_state_is_not_pushed_back(self):
        plant, scenario, config = build_plant()
        crossing_state = UAVPhysicalState(np.array([1.02, 0.75, 1.0]), np.array([0.12, 0.0, 0.0]), 100.0, 0.0)
        crossing_world = StaticWorld(
            config.world_size,
            (AABBObstacle(np.array([1.03, 0.7, 0.8]), np.array([1.031, 0.8, 1.2])),),
        )
        crossing_scenario = replace(scenario, initial_state=crossing_state, world=crossing_world)
        plant = CertifiedSingleUAVPlantEnv(config, crossing_scenario)
        plant.reset(seed=3)
        _, _, terminated, _, info = plant.step(np.zeros(3))
        self.assertTrue(terminated)
        self.assertEqual(info["failure_reason"], "collision")
        self.assertGreater(plant.state.position[0], 1.031)
        self.assertAlmostEqual(plant.state.velocity[0], 0.12)

    def test_world_boundary_is_a_swept_collision(self):
        world = StaticWorld(np.array([1.0, 1.0, 1.0]))
        self.assertTrue(world.swept_collision(np.array([0.2, 0.5, 0.5]), np.array([0.01, 0.5, 0.5]), 0.05))


class EnergyLidarTerminalTests(unittest.TestCase):
    def test_realized_energy_increases_with_action(self):
        model = EnergyModel()
        state = UAVPhysicalState(np.ones(3), np.array([0.1, 0.0, 0.0]), 10.0, 0.0)
        self.assertGreaterEqual(model.realized_cost(state, np.ones(3), 0.2), model.realized_cost(state, np.zeros(3), 0.2))

    def test_energy_depletion_terminates_without_respawn_or_recharge(self):
        plant, scenario, config = build_plant()
        depleted_scenario = replace(
            scenario,
            initial_state=UAVPhysicalState(scenario.initial_state.position, scenario.initial_state.velocity, 0.001, 0.0),
        )
        plant = CertifiedSingleUAVPlantEnv(config, depleted_scenario)
        initial_position = plant.state.position.copy()
        plant.reset(seed=4)
        _, _, terminated, _, info = plant.step(np.zeros(3))
        self.assertTrue(terminated)
        self.assertEqual(info["failure_reason"], "energy_depleted")
        self.assertLessEqual(plant.state.energy, 0.0)
        self.assertFalse(np.allclose(plant.state.position, initial_position) and plant.state.energy == config.initial_energy)

    def test_realized_cost_is_below_selected_synthetic_upper_bound(self):
        runtime = make_certified_uav_env()
        runtime.reset(seed=5)
        state = runtime.plant.state
        action = runtime.config.a_max
        realized = runtime.plant.energy_model.realized_cost(state, action, runtime.config.dt)
        upper = runtime.calibration.energy.upper_cost(tuple(state.velocity), tuple(action))
        self.assertLessEqual(realized, upper)

    def test_lidar_has_32_rays_and_distinguishes_no_hit_from_invalid(self):
        world = StaticWorld(np.array([10.0, 10.0, 2.0]))
        model = HorizontalLidarModel(32, 0.1, "sensor-test")
        state = UAVPhysicalState(np.array([5.0, 5.0, 1.0]), np.zeros(3), 10.0, 1.0)
        packet = model.measure(state, world, np.random.default_rng(6))
        self.assertEqual(packet.distances.shape, (32,))
        self.assertTrue(np.all(packet.valid))
        self.assertFalse(np.any(packet.hit))
        model.forced_invalid_indices.add(0)
        invalid_packet = model.measure(state, world, np.random.default_rng(6))
        self.assertFalse(invalid_packet.valid[0])
        self.assertFalse(invalid_packet.hit[0])
        self.assertEqual(invalid_packet.sensor_version, "sensor-test")

    def test_terminal_requires_position_velocity_energy_and_hover_evidence(self):
        base = TerminalSpec(np.zeros(3), np.ones(3), np.full(3, 0.1), 1.0, ("hover",), "terminal")
        admissible = UAVPhysicalState(np.full(3, 0.5), np.zeros(3), 2.0, 0.0)
        self.assertTrue(base.is_admissible(admissible))
        self.assertFalse(base.is_admissible(UAVPhysicalState(np.full(3, 0.5), np.full(3, 0.2), 2.0, 0.0)))
        self.assertFalse(base.is_admissible(UAVPhysicalState(np.full(3, 0.5), np.zeros(3), 0.5, 0.0)))
        without_hover = replace(base, continuation_modes=())
        self.assertFalse(without_hover.is_admissible(admissible))
        self.assertNotIn("docking", base.continuation_modes)


class RuntimeAndScenarioTests(unittest.TestCase):
    def test_open_corridor_closes_and_executes_affine_tanh_candidate(self):
        runtime = make_certified_uav_env()
        observation, reset_info = runtime.reset(seed=7)
        self.assertTrue(reset_info["certificate_ready"])
        self.assertTrue(runtime.last_manifest.verify_integrity())
        actor_output = np.array([0.2, -0.3, 0.1])
        _, _, _, _, info = runtime.step(actor_output)
        trace = info["telemetry"].action_trace
        self.assertTrue(trace.accepted)
        record = runtime.replay.records[-1]
        expected = np.asarray(record.zonotope_center) + np.asarray(record.zonotope_generators) @ np.tanh(actor_output)
        np.testing.assert_allclose(trace.candidate, expected)
        np.testing.assert_allclose(trace.published, trace.candidate)
        np.testing.assert_allclose(info["critic_action"], trace.published)
        self.assertEqual(observation.shape, runtime.observation_space.shape)

    def test_actor_nan_falls_back_and_is_never_published(self):
        runtime = make_certified_uav_env()
        runtime.reset(seed=8)
        _, _, _, _, info = runtime.step(np.array([np.nan, 0.0, 0.0]))
        trace = info["telemetry"].action_trace
        self.assertFalse(trace.accepted)
        self.assertEqual(trace.fallback_reason, "CERTIFIER_EXCEPTION")
        np.testing.assert_allclose(trace.published, trace.fallback)

    def test_narrow_scenario_has_certified_kappa_but_no_generator(self):
        runtime = make_certified_uav_env("narrow_corridor.json")
        _, reset_info = runtime.reset(seed=9)
        self.assertFalse(reset_info["certificate_ready"])
        self.assertEqual(reset_info["certificate_failure_reason"], "NO_GENERATOR_SET")
        calls = runtime.actor.calls
        _, _, _, _, info = runtime.step(np.zeros(3))
        self.assertEqual(runtime.actor.calls, calls)
        self.assertFalse(info["accepted"])

    def test_invalid_corridor_skips_actor_and_fails_closed(self):
        runtime = make_certified_uav_env("invalid_corridor.json")
        _, reset_info = runtime.reset(seed=10)
        self.assertEqual(reset_info["certificate_failure_reason"], "INITIAL_STATE_OUTSIDE_CORRIDOR_SUFFIX")
        calls = runtime.actor.calls
        _, _, _, _, info = runtime.step(np.zeros(3))
        self.assertEqual(runtime.actor.calls, calls)
        self.assertFalse(info["accepted"])

    def test_insufficient_energy_rejects_task_mode(self):
        runtime = make_certified_uav_env("insufficient_energy.json")
        _, reset_info = runtime.reset(seed=11)
        self.assertEqual(reset_info["certificate_failure_reason"], "INSUFFICIENT_RECOVERY_RESERVE")

    def test_insufficient_sensing_refuses_task_mode(self):
        scenario = FixedCertificationScenario().definition
        config = CertifiedUAVConfig(world_size=scenario.world_size, lidar_range=0.01)
        plant = CertifiedSingleUAVPlantEnv(config, scenario)
        runtime = __import__("envs.certified_uav", fromlist=["CertifiedRuntimeWrapper"]).CertifiedRuntimeWrapper(CertifiedTaskWrapper(plant))
        _, info = runtime.reset(seed=12)
        self.assertEqual(info["certificate_failure_reason"], "INSUFFICIENT_SENSING_FOR_BRAKING_TUBE")

    def test_watchdog_deadline_keeps_kappa_published(self):
        runtime = make_certified_uav_env()
        runtime.reset(seed=13)
        runtime.watchdog = SimulatedWatchdog(0.0, WCETContract(control_period_seconds=runtime.config.dt))
        _, _, _, _, info = runtime.step(np.zeros(3))
        self.assertFalse(info["accepted"])
        self.assertEqual(info["fallback_reason"], "WATCHDOG_DEADLINE")
        self.assertEqual(runtime.watchdog.last_trace.publication_count, 1)

    def test_certificate_version_mutation_rejects_candidate(self):
        runtime = make_certified_uav_env()
        runtime.reset(seed=14)
        original = runtime.actor.sample_u

        def mutating_actor(observation):
            runtime.geometry.version += 1
            return original(observation)

        runtime.actor.sample_u = mutating_actor
        _, _, _, _, info = runtime.step(np.zeros(3))
        self.assertFalse(info["accepted"])
        self.assertEqual(info["fallback_reason"], "CERTIFIER_EXCEPTION")

    def test_replay_and_calibration_log_separate_actions_and_versions(self):
        runtime = make_certified_uav_env()
        runtime.reset(seed=15)
        runtime.step(np.array([0.1, 0.0, -0.1]))
        record = runtime.replay.records[-1]
        telemetry = runtime.plant.last_telemetry
        self.assertEqual(record.critic_action, record.executed_action)
        self.assertEqual(record.certificate_version, record.certificate_state.certificate_version)
        self.assertEqual(tuple(telemetry.action_trace.published), record.executed_action)
        exported = runtime.export_calibration_record()[-1]
        self.assertEqual(exported["evidence_kind"], "synthetic-simulator")
        self.assertEqual(exported["published_action"], tuple(telemetry.action_trace.published))

    def test_runtime_source_does_not_read_plant_world(self):
        from envs.certified_uav.runtime_wrapper import CertifiedRuntimeWrapper

        source = inspect.getsource(CertifiedRuntimeWrapper)
        self.assertNotIn("plant.world", source)
        self.assertNotIn("scenario.world", source)


class CompatibilityTests(unittest.TestCase):
    def test_legacy_alias_preserves_original_classes_and_step(self):
        from envs.UAVEnergyDelivery import UAVEnvDiscreteWrapper as Original
        from envs.legacy.multi_uav_delivery_env import UAVEnvDiscreteWrapper as LegacyAlias

        self.assertIs(Original, LegacyAlias)
        environment = LegacyAlias(num_hunters=2, num_obstacle=1, episode_limit=3, total_orders=2, max_active_orders=1)
        observation = environment.reset()
        self.assertIsInstance(observation, np.ndarray)
        result = environment.step(np.zeros((2, 3), dtype=np.float32))
        self.assertEqual(len(result), 3)

    def test_new_environment_uses_distinct_class_names(self):
        runtime = make_certified_uav_env()
        self.assertEqual(type(runtime).__name__, "CertifiedRuntimeWrapper")
        self.assertEqual(type(runtime.plant).__name__, "CertifiedSingleUAVPlantEnv")


if __name__ == "__main__":
    unittest.main()
