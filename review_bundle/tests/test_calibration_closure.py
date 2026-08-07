from __future__ import annotations

import math
import unittest
from dataclasses import replace

from calibration.confidence import BoundEstimate
from calibration.dynamics import DynamicsSample, build_dynamics_contract
from calibration.energy import EnergySample, build_energy_contract
from calibration.schema import (
    ConfidenceSemantics,
    EvidenceMetadata,
    OperatingDomain,
    SourceKind,
    DataSplit,
    evidence_hash,
)
from calibration.sensor import build_sensor_contract
from calibration.tracking import TrackingSample, build_tracking_contract
from calibration.validation import CalibrationRegistry
from calibration.synthetic import build_synthetic_calibration_bundle, synthetic_metadata
from calibration.terminal import build_terminal_contract
from cert_runtime.closed_loop import DeterministicClosedLoopHarness, FailureInjection
from cert_runtime.geometry import CellState, LidarRay, RollingLocalGeometry, SensorBounds
from cert_runtime.synthetic import (
    build_synthetic_closure_fixture,
    synthetic_adapters,
    synthetic_watchdog,
)
from cert_runtime.types import AABB2
from cert_runtime.wcet import WCETBenchmarkHarness
from cert_runtime.contracts import WCETContract
from cert_runtime.invalidation import dependency_invalidation_plan


class CalibrationContractTests(unittest.TestCase):
    def test_missing_evidence_id_is_rejected(self):
        metadata = synthetic_metadata("valid", "v1")
        with self.assertRaises(ValueError):
            replace(metadata, evidence_id="")

    def test_invalid_operating_domain_is_rejected(self):
        with self.assertRaises(ValueError):
            OperatingDomain((2.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), ("hover",))

    def test_expired_or_version_mismatched_contract_is_rejected(self):
        contracts, _ = build_synthetic_calibration_bundle()
        sensor = contracts[0]
        point = {
            "speed": 0.5,
            "acceleration": 0.5,
            "payload": 0.5,
            "temperature": 20.0,
            "voltage": 20.0,
            "flight_mode": "hover",
        }
        self.assertFalse(sensor.is_applicable(1001.0, point, sensor.metadata.device_version, sensor.metadata.firmware_version))
        self.assertFalse(sensor.is_applicable(20.0, point, "wrong-device", sensor.metadata.firmware_version))

    def test_empirical_quantile_cannot_be_labeled_deterministic(self):
        contracts, _ = build_synthetic_calibration_bundle()
        sensor = contracts[0]
        records = []
        from calibration.schema import DataSplit, RawCalibrationRecord

        point = tuple(sorted({
            "speed": 0.5,
            "acceleration": 0.5,
            "payload": 0.5,
            "temperature": 20.0,
            "voltage": 20.0,
            "flight_mode": "hover",
        }.items()))
        for channel in ("position", "attitude", "range", "time_sync"):
            for index, split in enumerate((DataSplit.CALIBRATION, DataSplit.VALIDATION)):
                records.append(RawCalibrationRecord(
                    f"{channel}-{index}", index + 1.0, channel, 0.01, 0.0, split,
                    sensor.metadata.device_version, sensor.metadata.firmware_version, point,
                ))
        metadata = replace(
            sensor.metadata,
            confidence_semantics=ConfidenceSemantics.DETERMINISTIC_ENGINEERING,
            confidence_delta=None,
            simultaneous_family_size=None,
        )
        estimates = {
            name: BoundEstimate(0.02, ConfidenceSemantics.EMPIRICAL_QUANTILE, None, ("finite sample",), False)
            for name in ("position", "attitude", "range", "time_sync")
        }
        with self.assertRaises(ValueError):
            build_sensor_contract(
                records,
                metadata,
                "invalid-deterministic",
                estimates,
                beam_half_angle_radians=0.2,
                footprint_radius=0.1,
                map_discretization_error=0.1,
                maximum_range=5.0,
                maximum_speed=2.0,
                evidence_max_age_seconds=1.0,
                minimum_free_observations=1,
            )

    def test_synthetic_reports_remain_blocked_by_calibration(self):
        _, reports = build_synthetic_calibration_bundle()
        self.assertTrue(all(report.physical_status == "blocked-by-calibration" for report in reports))

    def test_sensor_version_change_invalidates_free_cells(self):
        geometry = RollingLocalGeometry(-2.0, -2.0, 8, 8, 0.5)
        geometry.mark_free_from_certificate(AABB2(-1.0, -1.0, 1.0, 1.0), "proof", 1.0, "sensor-v1")
        bounds = SensorBounds(
            0.01, 0.01, 0.01, 0.2, 0.01, 0.05, 0.01, 4.0, 2.0, 1.0, 1, "sensor-v2"
        )
        geometry.update_lidar((0.0, 0.0), (LidarRay(0.0, 1.0, False, 2.0),), bounds, 2.0)
        self.assertEqual(geometry.active_calibration_version, "sensor-v2")
        self.assertTrue(
            all(
                geometry.state_at(row, column) != CellState.FREE
                for row in range(geometry.height)
                for column in range(geometry.width)
            )
        )

    def test_out_of_domain_velocity_blocks_bundle(self):
        fixture = build_synthetic_closure_fixture()
        point = dict(fixture.operating_point)
        point["speed"] = 9.0
        valid, reason = fixture.calibration.validate(
            fixture.timestamp,
            point,
            fixture.device_version,
            fixture.firmware_version,
            allow_synthetic=True,
        )
        self.assertFalse(valid)
        self.assertIn("out-of-domain", reason)

    def test_terminal_claim_requires_continuation_evidence(self):
        metadata = synthetic_metadata("terminal", "terminal-v2")
        with self.assertRaises(ValueError):
            build_terminal_contract(
                metadata,
                "terminal-v2",
                horizontal_position=(-1.0, -1.0, 1.0, 1.0),
                altitude=(-0.2, 0.2),
                velocity_low=(-0.1, -0.1, -0.1),
                velocity_high=(0.1, 0.1, 0.1),
                minimum_energy=1.0,
                continuation_evidence=(),
            )

    def test_tracking_validation_exceedance_is_reported(self):
        metadata = synthetic_metadata("tracking-exceedance", "tracking-bad")
        point = tuple(sorted({
            "speed": 0.5, "acceleration": 0.5, "payload": 0.5,
            "temperature": 20.0, "voltage": 20.0, "flight_mode": "hover",
        }.items()))
        samples = (
            TrackingSample("cal", 0.0, 0.01, 0.02, (0.0,)*3, (0.0,)*3, (0.01,)*3, DataSplit.CALIBRATION, point),
            TrackingSample("val", 1.0, 1.01, 1.02, (0.0,)*3, (0.0,)*3, (0.2,)*3, DataSplit.VALIDATION, point),
        )
        contract, report = build_tracking_contract(samples, metadata, "tracking-bad", (0.05,)*3, 0.02)
        self.assertFalse(report.valid)
        self.assertEqual(contract.status, "blocked-by-calibration")
        self.assertTrue(all(channel.validation_exceedances == 1 for channel in report.channels))

    def test_dynamics_validation_exceedance_is_reported(self):
        contracts, _ = build_synthetic_calibration_bundle()
        tracking = contracts[1]
        metadata = synthetic_metadata("dynamics-exceedance", "dynamics-bad")
        point = tuple(sorted({
            "speed": 0.5, "acceleration": 0.5, "payload": 0.5,
            "temperature": 20.0, "voltage": 20.0, "flight_mode": "hover",
        }.items()))
        samples = (
            DynamicsSample("cal", 0.0, 1.0, (0.0,)*3, (0.0,)*3, (0.0,)*3, (0.0,)*3, (0.0,)*3, (0.0,)*3, (0.0,)*3, DataSplit.CALIBRATION, point),
            DynamicsSample("val", 1.0, 2.0, (0.0,)*3, (0.0,)*3, (1.0,)*3, (1.0,)*3, (0.0,)*3, (0.0,)*3, (0.0,)*3, DataSplit.VALIDATION, point),
        )
        contract, report = build_dynamics_contract(
            samples, metadata, "dynamics-bad", tracking,
            initial_position_radius=(0.01,)*3,
            initial_velocity_radius=(0.01,)*3,
            control_period=1.0,
            control_period_error=0.01,
            sensor_latency_upper=0.01,
            compute_latency_upper=0.01,
            switch_latency_upper=0.01,
            position_residual_radius=(0.1,)*3,
            velocity_residual_radius=(0.1,)*3,
            wind_acceleration_radius=(0.01,)*3,
        )
        self.assertFalse(report.valid)
        self.assertEqual(contract.status, "blocked-by-calibration")
        self.assertGreater(sum(channel.validation_exceedances for channel in report.channels), 0)

    def test_energy_underestimation_is_reported(self):
        metadata = synthetic_metadata("energy-exceedance", "energy-bad")
        samples = (
            EnergySample("cal", 0.0, 1.0, 20.0, 20.0, 0.1, 0.1, 0.1, (0.0,)*3, (0.0,)*3, False, 0.0, 20.0, 0.5, DataSplit.CALIBRATION),
            EnergySample("val", 1.0, 2.0, 20.0, 20.0, 0.1, 0.1, 5.0, (0.0,)*3, (0.0,)*3, False, 0.0, 20.0, 0.5, DataSplit.VALIDATION),
        )
        contract, report = build_energy_contract(
            samples, metadata, "energy-bad",
            avionics_cost=0.01, hover_cost=0.01,
            velocity_coefficients=(0.0,)*3, action_coefficients=(0.0,)*3,
            communication_cost=0.0, computation_cost=0.0,
            measurement_error=0.01, underestimation_margin=0.01,
        )
        self.assertFalse(report.valid)
        self.assertEqual(contract.status, "blocked-by-calibration")
        self.assertGreater(report.maximum_underestimation, 0.0)

    def test_same_version_different_evidence_is_rejected(self):
        registry = CalibrationRegistry()
        registry.register("energy", "v1", "hash-a")
        registry.register("energy", "v1", "hash-a")
        with self.assertRaises(ValueError):
            registry.register("energy", "v1", "hash-b")

    def test_bound_version_change_invalidates_dependent_objects(self):
        plan = dependency_invalidation_plan(
            {"energy": "v1", "dynamics": "d1"},
            {"energy": "v2", "dynamics": "d1"},
        )
        self.assertIn("recovery-energy", plan.invalidated_objects)
        self.assertIn("zonotope", plan.invalidated_objects)
        self.assertNotIn("geometry", plan.invalidated_objects)

    def test_each_sensor_uncertainty_contributes_to_hit_dilation(self):
        ray = LidarRay(1.0, 0.0, 2.0, True, True, "scan", 1.0)

        def occupied_count(**overrides):
            values = dict(
                position_error=0.0,
                attitude_error_radians=0.0,
                range_error=0.0,
                beam_half_angle_radians=0.01,
                time_sync_error=0.0,
                footprint_radius=0.0,
                map_discretization_error=0.0,
                maximum_range=5.0,
                maximum_speed=2.0,
                evidence_max_age_seconds=1.0,
                minimum_free_observations=1,
                calibration_version="sensor-dilation",
            )
            values.update(overrides)
            grid = RollingLocalGeometry(-3.0, -3.0, 24, 24, 0.25)
            grid.update_lidar((0.0, 0.0), (ray,), SensorBounds(**values), 1.0)
            return sum(
                grid.state_at(row, column) == CellState.OCCUPIED
                for row in range(grid.height)
                for column in range(grid.width)
            )

        baseline = occupied_count()
        variants = (
            {"position_error": 0.3},
            {"attitude_error_radians": 0.2},
            {"range_error": 0.3},
            {"beam_half_angle_radians": 0.3},
            {"time_sync_error": 0.2},
            {"footprint_radius": 0.3},
            {"map_discretization_error": 0.3},
        )
        for parameters in variants:
            with self.subTest(parameters=parameters):
                self.assertGreater(occupied_count(**parameters), baseline)


class SingleCorridorClosureTests(unittest.TestCase):
    def test_fixed_corridor_generates_complete_linked_manifest(self):
        fixture = build_synthetic_closure_fixture()
        result = fixture.closure.close(
            fixture.state,
            fixture.geometry,
            fixture.corridor,
            fixture.cells,
            fixture.operating_point,
            fixture.device_version,
            fixture.firmware_version,
            fixture.timestamp,
            allow_synthetic=True,
        )
        self.assertTrue(result.closed)
        self.assertEqual(result.status, "conditionally-verified-blocked-by-calibration")
        self.assertIsNotNone(result.manifest)
        self.assertEqual(result.manifest.manifest_hash, result.manifest.expected_hash)
        self.assertTrue(result.manifest.complete)
        self.assertGreaterEqual(len(result.manifest.entries), 11)
        self.assertTrue(all(cell.recovery_certificate is not None for cell in fixture.corridor.cells))
        self.assertTrue(all(cell.energy_certificate is not None for cell in fixture.corridor.cells))

    def test_one_failed_cell_breaks_entire_corridor_and_reports_witness(self):
        fixture = build_synthetic_closure_fixture()
        bad_outer = replace(
            fixture.cells[1],
            state_bounds=replace(
                fixture.cells[1].state_bounds,
                velocity=type(fixture.cells[1].state_bounds.velocity)(
                    (-5.0, -5.0, -5.0),
                    (5.0, 5.0, 5.0),
                ),
            ),
            maximum_speed=10.0,
        )
        result = fixture.closure.close(
            fixture.state,
            fixture.geometry,
            fixture.corridor,
            (fixture.cells[0], bad_outer),
            fixture.operating_point,
            fixture.device_version,
            fixture.firmware_version,
            fixture.timestamp,
            allow_synthetic=True,
        )
        self.assertFalse(result.closed)
        self.assertIsNotNone(result.failure_witness)
        self.assertIsNone(result.manifest)

    def test_manifest_tamper_is_detectable(self):
        fixture = build_synthetic_closure_fixture()
        result = fixture.closure.close(
            fixture.state, fixture.geometry, fixture.corridor, fixture.cells,
            fixture.operating_point, fixture.device_version, fixture.firmware_version,
            fixture.timestamp, allow_synthetic=True,
        )
        tampered = replace(result.manifest, complete=False)
        self.assertNotEqual(tampered.manifest_hash, tampered.expected_hash)

    def test_real_mode_rejects_synthetic_calibration(self):
        fixture = build_synthetic_closure_fixture()
        result = fixture.closure.close(
            fixture.state, fixture.geometry, fixture.corridor, fixture.cells,
            fixture.operating_point, fixture.device_version, fixture.firmware_version,
            fixture.timestamp, allow_synthetic=False,
        )
        self.assertFalse(result.closed)
        self.assertIn("blocked-by-calibration", result.failure_witness.failed_predicate)


class WCETAndClosedLoopTests(unittest.TestCase):
    def test_desktop_benchmark_is_profiling_not_hard_wcet(self):
        contract = WCETContract(control_period_seconds=0.1, margin_seconds=0.01)
        report = WCETBenchmarkHarness(contract).run(
            {"sensor": lambda size: sum(range(size)), "publish": lambda size: size + 1},
            (1, 10),
            warmup_runs=1,
            measured_runs=3,
        )
        self.assertFalse(report.deployment_qualified)
        self.assertEqual(report.status, "blocked-by-deployment-evidence")
        self.assertTrue(all(math.isfinite(value) for _, value in report.per_stage_maxima))

    def _harness(self):
        fixture = build_synthetic_closure_fixture()
        adapters = synthetic_adapters(fixture)
        harness = DeterministicClosedLoopHarness(
            fixture.closure,
            fixture.runtime,
            synthetic_watchdog(),
            adapters["state"],
            adapters["lidar"],
            adapters["energy"],
            adapters["sink"],
            adapters["time"],
            adapters["recorder"],
            fixture.cells,
            fixture.operating_point,
            fixture.device_version,
            fixture.firmware_version,
            fixture.sensor_bounds,
        )
        return fixture, adapters, harness

    def test_normal_synthetic_cycle_publishes_only_complete_candidate(self):
        _, adapters, harness = self._harness()
        record = harness.run_cycle((0.0, 0.0, 0.0))
        self.assertTrue(record.accepted)
        self.assertEqual(record.command_source, "task")
        self.assertEqual(len(adapters["sink"].commands), 1)
        self.assertEqual(len(harness.runtime.replay.records), 1)
        self.assertEqual(harness.runtime.replay.records[0].critic_action, record.executed_action)
        self.assertTrue(harness.watchdog.last_trace.kappa_staged_before_worker)

    def test_out_of_domain_calibration_blocks_before_geometry_update(self):
        fixture, adapters, harness = self._harness()
        harness.operating_point = dict(harness.operating_point)
        harness.operating_point["speed"] = 9.0
        version_before = fixture.geometry.version
        record = harness.run_cycle((0.0, 0.0, 0.0))
        self.assertFalse(record.accepted)
        self.assertEqual(record.closure_status, "calibration-invalid")
        self.assertEqual(fixture.geometry.version, version_before)

    def test_all_failure_injections_fail_closed_and_are_logged(self):
        for injection in FailureInjection:
            with self.subTest(injection=injection.value):
                _, adapters, harness = self._harness()
                record = harness.run_cycle((0.0, 0.0, 0.0), injection)
                self.assertFalse(record.accepted)
                self.assertNotEqual(record.command_source, "task")
                self.assertEqual(record.injection, injection.value)
                self.assertEqual(len(adapters["recorder"].records), 1)


if __name__ == "__main__":
    unittest.main()
