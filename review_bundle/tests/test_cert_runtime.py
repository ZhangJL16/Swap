from __future__ import annotations

import math
import random
import time
import unittest
from dataclasses import replace

from cert_runtime import (
    AABB2,
    AtomicCommandPublisher,
    CandidateBundle,
    CellState,
    CertificateConfig,
    CertificateReplay,
    CertificateState,
    CorridorCell,
    CorridorRecoveryVerifier,
    DynamicsBounds,
    EnergyBounds,
    FrozenRecoveryPolicy,
    Interval,
    Interval3,
    LidarRay,
    RecoveryConfig,
    RecoveryEnergySolver,
    ReturnCorridor,
    RollingLocalGeometry,
    RuntimeCertifier,
    SensorBounds,
    RuntimeSensorBoundsContract,
    SimulatedWatchdog,
    StateCellBounds,
    SuccessorEnvelopeBuilder,
    TerminalCondition,
    WCETContract,
    Zonotope3,
    ZonotopeConstructor,
)


class FixedActor:
    def __init__(self, output=(0.2, -0.3, 0.5)) -> None:
        self.output = output
        self.calls = 0

    def sample_u(self, observation):
        self.calls += 1
        return self.output


class FailingActor:
    def __init__(self) -> None:
        self.calls = 0

    def sample_u(self, observation):
        self.calls += 1
        raise AssertionError("task actor must not be called")


class NonfiniteActor:
    def sample_u(self, observation):
        return (0.0, float("nan"), 0.0)


class SlowActor:
    def __init__(self, delay: float) -> None:
        self.delay = delay

    def sample_u(self, observation):
        time.sleep(self.delay)
        return (0.0, 0.0, 0.0)


class ConstantClock:
    def __init__(self, value=10.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class IncrementingClock:
    def __init__(self, increment: float, start: float = 10.0) -> None:
        self.value = start
        self.increment = increment

    def __call__(self) -> float:
        self.value += self.increment
        return self.value


def sensor_bounds(**overrides) -> SensorBounds:
    values = dict(
        position_error=0.02,
        attitude_error_radians=0.01,
        range_error=0.02,
        beam_half_angle_radians=0.35,
        time_sync_error=0.05,
        footprint_radius=0.03,
        map_discretization_error=0.01,
        maximum_range=5.0,
        maximum_speed=1.0,
        evidence_max_age_seconds=1.0,
        minimum_free_observations=1,
        calibration_version="synthetic-sensor-v1",
    )
    values.update(overrides)
    return SensorBounds(**values)


def build_certified_fixture(
    actor=None,
    minimum_sigma: float = 0.05,
    maximum_condition_number: float = 20.0,
    deadline_seconds: float = 1.0,
    clock=None,
    energy_calibrated: bool = True,
):
    reference_clock = clock or ConstantClock()
    geometry = RollingLocalGeometry(-6.0, -6.0, 24, 24, 0.5)
    geometry.mark_free_from_certificate(AABB2(-5.5, -5.5, 5.5, 5.5), "synthetic-free-proof", 10.0)
    corridor = ReturnCorridor(transfer_radius=0.1, geometry_margin=0.02)
    terminal_bounds = StateCellBounds(
        Interval3((-3.0, -3.0, -0.3), (3.0, 3.0, 0.3)),
        Interval3((-1.5, -1.5, -0.5), (1.5, 1.5, 0.5)),
        Interval(10.0, 101.0),
    )
    outer_bounds = StateCellBounds(
        Interval3((0.9, -0.1, -0.1), (1.1, 0.1, 0.1)),
        Interval3((-0.05, -0.05, -0.05), (0.05, 0.05, 0.05)),
        Interval(20.0, 100.0),
    )
    cells = [
        CorridorCell(0, AABB2(-4.0, -4.0, 4.0, 4.0), 2.5, geometry.version, terminal_bounds),
        CorridorCell(1, AABB2(0.5, -0.5, 1.5, 0.5), 2.0, geometry.version, outer_bounds),
    ]
    assert corridor.create(cells, geometry)
    dynamics = DynamicsBounds(
        control_period=0.5,
        position_radius=(0.01, 0.01, 0.01),
        velocity_radius=(0.01, 0.01, 0.01),
        acceleration_tracking_radius=(0.01, 0.01, 0.01),
        control_period_error=0.01,
        latency_upper=0.02,
        wind_acceleration_radius=(0.01, 0.01, 0.01),
        version="synthetic-dynamics-v1",
        calibration_complete=True,
    )
    energy = EnergyBounds(
        0.1,
        (0.01, 0.01, 0.02),
        0.05,
        additive_error_radius=0.01,
        version="synthetic-energy-v1",
        calibration_complete=energy_calibrated,
    )
    envelope_builder = SuccessorEnvelopeBuilder(dynamics, energy)
    recovery = FrozenRecoveryPolicy(
        RecoveryConfig(
            1.0,
            0.5,
            (2.0, 2.0, 2.0),
            0.25,
            1.0,
            0.5,
            braking_deceleration=10.0,
            update_latency=0.02,
            geometry_margin=0.02,
            parameter_version="synthetic-kappa-v1",
        )
    )
    terminal = TerminalCondition(
        AABB2(-5.0, -5.0, 5.0, 5.0),
        Interval(-1.0, 1.0),
        Interval3((-3.0, -3.0, -2.0), (3.0, 3.0, 2.0)),
        1.0,
        True,
        False,
        False,
        "synthetic-terminal-v1",
    )
    verifier = CorridorRecoveryVerifier(recovery, envelope_builder, terminal, 100.0)
    recovery_result = verifier.verify(corridor, geometry, timestamp=10.0)
    assert recovery_result.verified, recovery_result
    energy_solver = RecoveryEnergySolver(energy, 100.0)
    energy_result = energy_solver.solve(corridor, timestamp=10.0)
    state = CertificateState(
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        50.0,
        (0.0, 0.0, 0.0),
        geometry,
        corridor,
        position_error_radius=(0.01, 0.01, 0.01),
        velocity_error_radius=(0.01, 0.01, 0.01),
        energy_error_radius=0.01,
    )
    config = CertificateConfig(
        (-2.0, -2.0, -2.0),
        (2.0, 2.0, 2.0),
        minimum_sigma,
        maximum_condition_number,
        1.0,
        0.5,
        10.0,
        0.02,
        0.02,
        1e-8,
        deadline_seconds,
        8,
    )
    constructor = ZonotopeConstructor(
        envelope_builder,
        config,
        recovery.config.parameter_version,
        clock=reference_clock,
    )
    replay = CertificateReplay()
    selected_actor = actor or FixedActor()
    runtime = RuntimeCertifier(selected_actor, recovery, constructor, replay)
    return {
        "state": state,
        "geometry": geometry,
        "corridor": corridor,
        "envelope": envelope_builder,
        "recovery": recovery,
        "recovery_verifier": verifier,
        "energy_solver": energy_solver,
        "energy_result": energy_result,
        "constructor": constructor,
        "replay": replay,
        "actor": selected_actor,
        "runtime": runtime,
        "clock": reference_clock,
        "terminal": terminal,
    }


class CalibrationAndGeometryTests(unittest.TestCase):
    def test_incomplete_sensor_contract_is_blocked(self):
        contract = RuntimeSensorBoundsContract(position_error=0.1)
        self.assertEqual(contract.status, "blocked-by-calibration")
        with self.assertRaises(RuntimeError):
            SensorBounds.from_contract(contract)

    def test_invalid_sensor_contract_is_blocked(self):
        contract = RuntimeSensorBoundsContract(
            position_error=0.0,
            attitude_error_radians=2.0,
            range_error=0.0,
            beam_half_angle_radians=2.0,
            time_sync_error=0.0,
            footprint_radius=0.0,
            map_discretization_error=0.0,
            maximum_range=5.0,
            maximum_speed=1.0,
            evidence_max_age_seconds=1.0,
            minimum_free_observations=1,
            calibration_version="invalid-angular-contract",
        )
        self.assertEqual(contract.status, "blocked-by-calibration")
        with self.assertRaises(RuntimeError):
            SensorBounds.from_contract(contract)

    def test_unknown_is_default_and_invalid_or_max_range_rays_cannot_free(self):
        geometry = RollingLocalGeometry(-2.0, -2.0, 80, 80, 0.05)
        bounds = sensor_bounds()
        rays = [
            LidarRay(1.0, 0.0, 2.0, False, True, "f0", 1.0),
            LidarRay(1.0, 0.0, bounds.maximum_range, True, False, "f1", 1.0),
        ]
        geometry.update_lidar((0.0, 0.0), rays, bounds, 1.0)
        self.assertTrue(
            all(
                geometry.state_at(row, column) == CellState.UNKNOWN
                for row in range(geometry.height)
                for column in range(geometry.width)
            )
        )

    def test_only_fully_covered_cells_become_free_with_provenance(self):
        geometry = RollingLocalGeometry(-2.0, -2.0, 80, 80, 0.05)
        bounds = sensor_bounds()
        geometry.update_lidar(
            (0.0, 0.0),
            [LidarRay(1.0, 0.0, 2.0, True, True, "scan-1", 1.0)],
            bounds,
            1.0,
        )
        free_cells = [
            (row, column)
            for row in range(geometry.height)
            for column in range(geometry.width)
            if geometry.state_at(row, column) == CellState.FREE
        ]
        self.assertTrue(free_cells)
        for row, column in free_cells:
            evidence = geometry.evidence_at(row, column)
            self.assertTrue(evidence)
            self.assertEqual(evidence[0].sensor_frame, "scan-1")
            self.assertEqual(evidence[0].calibration_version, bounds.calibration_version)
        self.assertEqual(geometry.state_at(0, 0), CellState.UNKNOWN)

    def test_partially_covered_cells_do_not_become_free(self):
        geometry = RollingLocalGeometry(-1.0, -1.0, 8, 8, 0.25)
        bounds = sensor_bounds(
            beam_half_angle_radians=0.011,
            attitude_error_radians=0.01,
            footprint_radius=0.05,
        )
        geometry.update_lidar(
            (0.0, 0.0),
            [LidarRay(1.0, 0.0, 0.9, True, True, "narrow", 1.0)],
            bounds,
            1.0,
        )
        self.assertFalse(
            any(
                geometry.state_at(row, column) == CellState.FREE
                for row in range(geometry.height)
                for column in range(geometry.width)
            )
        )

    def test_boundary_and_historical_occupied_cells_do_not_become_free(self):
        geometry = RollingLocalGeometry(-1.0, -1.0, 8, 8, 0.25)
        geometry.mark_occupied_from_certificate(
            AABB2(0.5, -0.25, 0.75, 0.0),
            "static-obstacle-proof",
            0.5,
        )
        occupied_indices = [
            (row, column)
            for row in range(geometry.height)
            for column in range(geometry.width)
            if geometry.state_at(row, column) == CellState.OCCUPIED
        ]
        geometry.update_lidar(
            (0.0, 0.0),
            [LidarRay(1.0, 0.0, 0.95, True, True, "scan", 1.0)],
            sensor_bounds(),
            1.0,
        )
        self.assertTrue(occupied_indices)
        self.assertTrue(
            all(geometry.state_at(row, column) == CellState.OCCUPIED for row, column in occupied_indices)
        )
        boundary_indices = {
            (row, column)
            for row in range(geometry.height)
            for column in range(geometry.width)
            if row in (0, geometry.height - 1) or column in (0, geometry.width - 1)
        }
        self.assertTrue(
            all(geometry.state_at(row, column) != CellState.FREE for row, column in boundary_indices)
        )

    def test_error_dilation_increases_occupied_cover(self):
        small = RollingLocalGeometry(-3.0, -3.0, 24, 24, 0.25)
        large = RollingLocalGeometry(-3.0, -3.0, 24, 24, 0.25)
        ray = LidarRay(1.0, 0.0, 2.0, True, True, "scan", 1.0)
        small.update_lidar((0.0, 0.0), [ray], sensor_bounds(), 1.0)
        large.update_lidar(
            (0.0, 0.0),
            [ray],
            sensor_bounds(
                position_error=0.2,
                range_error=0.2,
                attitude_error_radians=0.08,
                footprint_radius=0.2,
            ),
            1.0,
        )
        count = lambda grid: sum(
            grid.state_at(row, column) == CellState.OCCUPIED
            for row in range(grid.height)
            for column in range(grid.width)
        )
        self.assertGreater(count(large), count(small))

    def test_stale_free_evidence_expires_to_unknown(self):
        geometry = RollingLocalGeometry(-2.0, -2.0, 80, 80, 0.05)
        bounds = sensor_bounds(evidence_max_age_seconds=0.5)
        geometry.update_lidar(
            (0.0, 0.0),
            [LidarRay(1.0, 0.0, 2.0, True, True, "scan", 1.0)],
            bounds,
            1.0,
        )
        self.assertTrue(
            any(
                geometry.state_at(r, c) == CellState.FREE
                for r in range(geometry.height)
                for c in range(geometry.width)
            )
        )
        old_version = geometry.version
        geometry.expire_stale(2.0, bounds.evidence_max_age_seconds)
        self.assertFalse(
            any(
                geometry.state_at(r, c) == CellState.FREE
                for r in range(geometry.height)
                for c in range(geometry.width)
            )
        )
        self.assertGreater(geometry.version, old_version)

    def test_recenter_preserves_only_identical_world_cells(self):
        geometry = RollingLocalGeometry(-2.0, -2.0, 8, 8, 0.5)
        geometry.mark_free_from_certificate(AABB2(-1.0, -1.0, 1.0, 1.0), "proof", 1.0)
        original_evidence = geometry.evidence_at(3, 3)
        geometry.recenter(0.5, 0.0)
        self.assertEqual(geometry.evidence_at(3, 2), original_evidence)
        self.assertEqual(geometry.state_at(3, 7), CellState.UNKNOWN)

    def test_geometry_version_invalidates_recovery_certificate(self):
        fixture = build_certified_fixture()
        state = fixture["state"]
        before = fixture["runtime"].recovery_decision(state, 10.0)
        self.assertTrue(before.certified)
        state.local_geometry.mark_free_from_certificate(AABB2(-0.25, -0.25, 0.25, 0.25), "new-version", 10.0)
        after = fixture["runtime"].recovery_decision(state, 10.0)
        self.assertFalse(after.certified)


class IntervalEnvelopeTests(unittest.TestCase):
    def test_outward_rounding_never_shrinks_exact_arithmetic(self):
        left = Interval(0.1, 0.2)
        right = Interval(0.3, 0.4)
        result = left + right
        self.assertLessEqual(result.low, 0.4)
        self.assertGreaterEqual(result.high, 0.6)
        product = left * right
        self.assertLessEqual(product.low, 0.03)
        self.assertGreaterEqual(product.high, 0.08)

    def test_full_action_interval_and_combined_errors_are_contained(self):
        fixture = build_certified_fixture()
        builder = fixture["envelope"]
        state = fixture["state"]
        action = Interval3((-1.2, -0.4, -0.2), (-0.8, 0.4, 0.2))
        envelope = builder.propagate_action_interval(state, action)
        generator = random.Random(11)
        for _ in range(200):
            dt = generator.uniform(0.49, 0.53)
            p = [state.position[i] + generator.uniform(-0.01, 0.01) for i in range(3)]
            v = [state.velocity[i] + generator.uniform(-0.01, 0.01) for i in range(3)]
            a = [
                generator.uniform(action.low[i], action.high[i])
                + generator.uniform(-0.02, 0.02)
                for i in range(3)
            ]
            p_next = [
                p[i] + dt * v[i] + 0.5 * dt * dt * a[i] + generator.uniform(-0.01, 0.01)
                for i in range(3)
            ]
            v_next = [v[i] + dt * a[i] + generator.uniform(-0.01, 0.01) for i in range(3)]
            self.assertTrue(envelope.position.contains_point(p_next))
            self.assertTrue(envelope.velocity.contains_point(v_next))

    def test_tracking_and_latency_expand_successor(self):
        fixture = build_certified_fixture()
        state = fixture["state"]
        full = fixture["envelope"].propagate_point_action(state, (-1.0, 0.0, 0.0))
        narrow_builder = SuccessorEnvelopeBuilder(
            replace(
                fixture["envelope"].dynamics,
                acceleration_tracking_radius=(0.0, 0.0, 0.0),
                wind_acceleration_radius=(0.0, 0.0, 0.0),
                latency_upper=0.0,
            ),
            fixture["envelope"].energy,
        )
        narrow = narrow_builder.propagate_point_action(state, (-1.0, 0.0, 0.0))
        self.assertGreater(full.position.components[0].width, narrow.position.components[0].width)
        self.assertGreater(full.velocity.components[0].width, narrow.velocity.components[0].width)

    def test_energy_lower_bound_is_conservative(self):
        fixture = build_certified_fixture()
        state = fixture["state"]
        action = Interval3((-2.0, -1.0, -0.5), (1.0, 1.5, 0.5))
        envelope = fixture["envelope"].propagate_action_interval(state, action)
        self.assertLess(envelope.energy_low, state.energy)
        self.assertGreaterEqual(envelope.energy_high, state.energy - state.energy_error_radius)

    def test_nan_and_infinity_fail_closed(self):
        with self.assertRaises(ValueError):
            Interval(float("nan"), 1.0)
        with self.assertRaises(ValueError):
            Zonotope3.diagonal((0.0, 0.0, 0.0), (0.1, float("inf"), 0.1))


class RecoveryAndEnergyCertificateTests(unittest.TestCase):
    def test_every_installed_cell_has_hashed_one_step_certificate(self):
        fixture = build_certified_fixture()
        for cell in fixture["corridor"].cells:
            certificate = cell.recovery_certificate
            self.assertIsNotNone(certificate)
            self.assertTrue(certificate.certificate_hash)
            if cell.cell_id == 0:
                self.assertEqual(certificate.progress_result, "terminal")
            else:
                self.assertEqual(certificate.progress_result, "one-step-lower-level")
                self.assertTrue(all(identifier < cell.cell_id for identifier in certificate.successor_cell_ids))

    def test_cell_without_continuous_descent_is_not_certified(self):
        fixture = build_certified_fixture()
        corridor = fixture["corridor"]
        corridor.invalidate_certificates()
        bad = replace(
            corridor.cells[1],
            state_bounds=StateCellBounds(
                Interval3((3.5, -0.1, -0.1), (3.9, 0.1, 0.1)),
                corridor.cells[1].state_bounds.velocity,
                corridor.cells[1].state_bounds.energy,
            ),
        )
        corridor.cells[1] = bad
        result = fixture["recovery_verifier"].verify(corridor, fixture["geometry"], 10.0)
        self.assertFalse(result.verified)
        self.assertEqual(result.failed_cell_id, 1)
        self.assertTrue(all(cell.recovery_certificate is None for cell in corridor.cells))

    def test_kappa_runtime_authority_does_not_call_task_actor(self):
        actor = FailingActor()
        fixture = build_certified_fixture(actor=actor, minimum_sigma=3.0)
        decision = fixture["runtime"].step(fixture["state"], [0.0])
        self.assertFalse(decision.accepted)
        self.assertTrue(decision.recovery_certified)
        self.assertEqual(actor.calls, 0)

    def test_tampered_recovery_certificate_is_rejected(self):
        fixture = build_certified_fixture()
        cell = fixture["corridor"].cells[1]
        tampered = replace(cell.recovery_certificate, transit_cost_upper=0.0)
        fixture["corridor"].cells[1] = replace(cell, recovery_certificate=tampered)
        decision = fixture["runtime"].recovery_decision(fixture["state"], 10.0)
        self.assertFalse(decision.certified)
        self.assertEqual(decision.reason, "stale-cell-certificate")

    def test_energy_backward_order_and_e3_residual(self):
        fixture = build_certified_fixture()
        result = fixture["energy_result"]
        self.assertTrue(result.verified)
        self.assertEqual(result.certificates[0].transit_energy_upper, 0.0)
        self.assertGreater(result.certificates[1].transit_energy_upper, 0.0)
        self.assertTrue(fixture["energy_solver"].verify_residuals(fixture["corridor"], result.certificates))

    def test_reducing_energy_upper_is_detected(self):
        fixture = build_certified_fixture()
        certificates = dict(fixture["energy_result"].certificates)
        certificates[1] = replace(
            certificates[1],
            transit_energy_upper=certificates[1].transit_energy_upper - 0.1,
        )
        self.assertFalse(fixture["energy_solver"].verify_residuals(fixture["corridor"], certificates))

    def test_terminal_energy_is_separate_from_transit_energy(self):
        fixture = build_certified_fixture()
        terminal_certificate = fixture["energy_result"].certificates[0]
        self.assertEqual(terminal_certificate.transit_energy_upper, 0.0)
        self.assertEqual(fixture["recovery"].config.terminal_energy, 1.0)

    def test_uncalibrated_energy_solver_is_blocked(self):
        fixture = build_certified_fixture(energy_calibrated=False)
        self.assertFalse(fixture["energy_result"].verified)
        self.assertEqual(fixture["energy_result"].status, "blocked-by-calibration")


class ZonotopeAndRuntimeTests(unittest.TestCase):
    def test_complete_action_interval_is_propagated_and_verified(self):
        fixture = build_certified_fixture()
        state = fixture["state"]
        recovery = fixture["runtime"].recovery_decision(state, 10.0)
        certificate = fixture["constructor"].construct(state, recovery, 10.0)
        self.assertTrue(certificate.verified)
        direct = fixture["envelope"].propagate_action_interval(
            state,
            certificate.zonotope.action_bounds,
        )
        self.assertEqual(certificate.successor_envelope.position, direct.position)
        self.assertIsNotNone(
            fixture["constructor"].verify_complete(state, certificate.zonotope, recovery, 10.0)
        )

    def test_random_actor_outputs_map_inside_single_state_level_certificate(self):
        fixture = build_certified_fixture()
        generator = random.Random(7)
        for _ in range(100):
            fixture["actor"].output = tuple(generator.uniform(-8.0, 8.0) for _ in range(3))
            decision = fixture["runtime"].step(fixture["state"], [0.0, 1.0])
            self.assertTrue(decision.accepted)
            self.assertTrue(decision.certificate.zonotope.contains(decision.action))

    def test_sigma_or_no_positive_volume_causes_kappa(self):
        actor = FailingActor()
        fixture = build_certified_fixture(actor=actor, minimum_sigma=3.0)
        decision = fixture["runtime"].step(fixture["state"], [0.0])
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.fallback_reason, "NO_GENERATOR_SET")
        self.assertEqual(actor.calls, 0)

    def test_condition_number_limit_rejects_ill_conditioned_candidate(self):
        fixture = build_certified_fixture(maximum_condition_number=2.0)
        state = fixture["state"]
        recovery = fixture["runtime"].recovery_decision(state, 10.0)
        ill_conditioned = Zonotope3.diagonal(recovery.action, (0.05, 0.05, 0.5))
        self.assertIsNone(
            fixture["constructor"].verify_complete(state, ill_conditioned, recovery, 10.0)
        )

    def test_insufficient_energy_reserve_rejects_task_mode(self):
        fixture = build_certified_fixture()
        fixture["state"].energy = 1.1
        decision = fixture["runtime"].step(fixture["state"], [0.0])
        self.assertFalse(decision.accepted)
        self.assertFalse(decision.recovery_certified)

    def test_version_change_prevents_old_candidate_execution(self):
        fixture = build_certified_fixture()

        class MutatingActor:
            def sample_u(self, observation):
                fixture["geometry"].mark_free_from_certificate(
                    AABB2(-0.25, -0.25, 0.25, 0.25), "version-change", 10.0
                )
                return (0.0, 0.0, 0.0)

        fixture["runtime"].actor = MutatingActor()
        decision = fixture["runtime"].step(fixture["state"], [0.0])
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.fallback_reason, "CERTIFICATE_VERSION_CHANGED")

    def test_deadline_expiry_prevents_actor_call(self):
        actor = FailingActor()
        fixture = build_certified_fixture(
            actor=actor,
            deadline_seconds=0.0,
            clock=IncrementingClock(0.1),
        )
        decision = fixture["runtime"].step(fixture["state"], [0.0])
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.fallback_reason, "DEADLINE")
        self.assertEqual(actor.calls, 0)

    def test_nonfinite_actor_output_fails_closed(self):
        fixture = build_certified_fixture(actor=NonfiniteActor())
        decision = fixture["runtime"].step(fixture["state"], [0.0])
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.fallback_reason, "ACTOR_NONFINITE")

    def test_replay_never_mixes_nominal_candidate_and_executed_actions(self):
        fixture = build_certified_fixture()
        decision = fixture["runtime"].step(fixture["state"], [3.0, 4.0])
        record = fixture["replay"].records[-1]
        self.assertTrue(decision.accepted)
        self.assertEqual(record.task_observation, (3.0, 4.0))
        self.assertNotEqual(record.nominal_pre_squash_u, record.executed_action)
        self.assertEqual(record.candidate_action, record.executed_action)
        self.assertEqual(record.critic_action, record.executed_action)
        self.assertEqual(record.certificate_version, fixture["state"].certificate_version)


class WatchdogTests(unittest.TestCase):
    def _watchdog_fixture(self):
        fixture = build_certified_fixture()
        state = fixture["state"]
        snapshot = state.snapshot()
        recovery = fixture["runtime"].recovery_decision(state, 10.0)
        watchdog = SimulatedWatchdog(0.02, WCETContract())
        return fixture, state, snapshot, recovery, watchdog

    def test_blocked_solver_still_publishes_kappa(self):
        fixture, state, snapshot, recovery, watchdog = self._watchdog_fixture()

        def blocked():
            time.sleep(0.1)
            return fixture["runtime"].prepare_candidate_bundle(state, [0.0], recovery, 10.0)

        command = watchdog.execute(snapshot, recovery.action, blocked, lambda: state.certificate_version)
        self.assertEqual(command.source, "kappa")
        self.assertEqual(command.reason, "WATCHDOG_DEADLINE")

    def test_certifier_exception_still_publishes_kappa(self):
        _, state, snapshot, recovery, watchdog = self._watchdog_fixture()

        def failure():
            raise RuntimeError("solver failed")

        command = watchdog.execute(snapshot, recovery.action, failure, lambda: state.certificate_version)
        self.assertEqual(command.source, "kappa")
        self.assertEqual(command.reason, "CERTIFIER_EXCEPTION")

    def test_actor_timeout_still_publishes_kappa(self):
        fixture, state, snapshot, recovery, watchdog = self._watchdog_fixture()
        fixture["runtime"].actor = SlowActor(0.1)
        command = watchdog.execute(
            snapshot,
            recovery.action,
            lambda: fixture["runtime"].prepare_candidate_bundle(state, [0.0], recovery, 10.0),
            lambda: state.certificate_version,
        )
        self.assertEqual(command.source, "kappa")
        self.assertEqual(command.reason, "WATCHDOG_DEADLINE")

    def test_only_complete_atomic_candidate_is_published(self):
        fixture, state, snapshot, recovery, watchdog = self._watchdog_fixture()
        command = watchdog.execute(
            snapshot,
            recovery.action,
            lambda: fixture["runtime"].prepare_candidate_bundle(state, [0.0], recovery, 10.0),
            lambda: state.certificate_version,
        )
        self.assertEqual(command.source, "task")

    def test_inconsistent_actor_bundle_is_rejected_atomically(self):
        fixture, state, snapshot, recovery, watchdog = self._watchdog_fixture()

        def producer():
            bundle = fixture["runtime"].prepare_candidate_bundle(state, [0.0], recovery, 10.0)
            return replace(bundle, eta=(0.5, 0.0, 0.0))

        command = watchdog.execute(
            snapshot,
            recovery.action,
            producer,
            lambda: state.certificate_version,
        )
        self.assertEqual(command.source, "kappa")
        self.assertEqual(command.reason, "STALE_OR_INCOMPLETE_BUNDLE")

    def test_stale_snapshot_is_forbidden(self):
        fixture, state, snapshot, recovery, watchdog = self._watchdog_fixture()

        def producer():
            bundle = fixture["runtime"].prepare_candidate_bundle(state, [0.0], recovery, 10.0)
            fixture["corridor"].certificate_epoch += 1
            return bundle

        command = watchdog.execute(snapshot, recovery.action, producer, lambda: state.certificate_version)
        self.assertEqual(command.source, "kappa")
        self.assertEqual(command.reason, "STALE_OR_INCOMPLETE_BUNDLE")

    def test_command_publication_occurs_only_once(self):
        fixture, state, snapshot, recovery, watchdog = self._watchdog_fixture()
        publisher = AtomicCommandPublisher()
        command = watchdog.execute(
            snapshot,
            recovery.action,
            lambda: fixture["runtime"].prepare_candidate_bundle(state, [0.0], recovery, 10.0),
            lambda: state.certificate_version,
            publisher,
        )
        self.assertEqual(command.source, "task")
        self.assertEqual(publisher.staged_default.source, "kappa")
        self.assertEqual(publisher.publication_count, 1)
        self.assertFalse(publisher.publish_once(command))

    def test_wcet_without_hardware_evidence_is_blocked(self):
        contract = WCETContract(
            sensor_seconds=0.001,
            update_seconds=0.001,
            kappa_seconds=0.001,
            set_construction_seconds=0.002,
            actor_seconds=0.001,
            publish_seconds=0.001,
            control_period_seconds=0.02,
        )
        self.assertEqual(contract.status, "blocked-by-deployment-evidence")


if __name__ == "__main__":
    unittest.main()
