from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cert_runtime.generator_sac import GeneratorReplayBuffer
from envs.certified_uav import make_certified_uav_env
from envs.certified_uav.scenario_families import generate_scenario_splits, scenario_file_hash
from experiments.agents import StatelessGeneratorPolicy
from tests.test_multistep_generator_sac import transition


class RLContributionAblationTests(unittest.TestCase):
    def test_center_only_uses_verified_center(self):
        runtime = make_certified_uav_env("mission_open.json", timing_mode="functional")
        observation, _ = runtime.reset(seed=0)
        context = runtime.action_context()
        self.assertTrue(context["generator_available"])
        policy = StatelessGeneratorPolicy("center_only", seed=0)
        _, _, _, _, info = runtime.step(policy.select_u(observation))
        trace = info["telemetry"].action_trace
        self.assertTrue(trace.accepted)
        np.testing.assert_allclose(trace.candidate, context["c"], atol=1e-10)

    def test_random_generator_remains_inside_verified_generator(self):
        runtime = make_certified_uav_env("mission_open.json", timing_mode="functional")
        observation, _ = runtime.reset(seed=1)
        context = runtime.action_context()
        u = StatelessGeneratorPolicy("random_generator", seed=1).select_u(observation)
        _, _, _, _, info = runtime.step(u)
        trace = info["telemetry"].action_trace
        self.assertTrue(trace.accepted)
        eta = np.linalg.solve(context["G"], trace.candidate - context["c"])
        self.assertTrue(np.all(np.abs(eta) <= 1.0 + 1e-12))

    def test_ablations_share_kappa_and_certificate(self):
        contexts = []
        for _ in ("center_only", "random_generator", "generator_sac", "shield_sac"):
            runtime = make_certified_uav_env("mission_open.json", timing_mode="functional")
            runtime.reset(seed=2)
            contexts.append(runtime.action_context())
        for context in contexts[1:]:
            np.testing.assert_allclose(context["kappa"], contexts[0]["kappa"])
            self.assertEqual(context["certificate_epoch"], contexts[0]["certificate_epoch"])
            self.assertEqual(context["recovery_hash"], contexts[0]["recovery_hash"])

    def test_center_modes_all_require_complete_set_verification(self):
        for mode in ("task_oriented", "zero", "braking"):
            runtime = make_certified_uav_env("mission_open.json", generator_center_mode=mode, timing_mode="functional")
            runtime.reset(seed=3)
            context = runtime.action_context()
            self.assertTrue(context["generator_available"], mode)
            self.assertIsNotNone(context["zonotope_hash"], mode)
            self.assertGreaterEqual(np.linalg.svd(context["G"], compute_uv=False).min(), runtime.config.minimum_generator_sigma - 1e-12)

    def test_intervention_categories_are_disjoint_by_phase(self):
        runtime = make_certified_uav_env("mission_open.json", timing_mode="functional")
        observation, _ = runtime.reset(seed=4)
        outbound_fallback = return_handoff = 0
        policy = StatelessGeneratorPolicy("center_only", seed=4)
        for _ in range(runtime.config.episode_limit):
            phase_before = runtime.task_env.phase.name
            observation, _, terminated, truncated, info = runtime.step(policy.select_u(observation))
            if not info["accepted"]:
                outbound_fallback += int(phase_before == "OUTBOUND")
                return_handoff += int(phase_before == "RETURN")
            if terminated or truncated:
                break
        self.assertEqual(outbound_fallback, 0)
        self.assertGreater(return_handoff, 0)


class ScenarioFamilyTests(unittest.TestCase):
    def test_split_generation_is_deterministic_and_disjoint(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            left = generate_scenario_splits(first, split_sizes={"training": 4, "validation": 2, "heldout": 4})
            right = generate_scenario_splits(second, split_sizes={"training": 4, "validation": 2, "heldout": 4})
            self.assertEqual([item.scenario_hash for item in left], [item.scenario_hash for item in right])
            identifiers = [item.scenario_id for item in left]
            self.assertEqual(len(identifiers), len(set(identifiers)))
            self.assertFalse({item.scenario_id for item in left if item.split == "training"} & {item.scenario_id for item in left if item.split == "heldout"})

    def test_scenario_hash_and_manifest_are_both_bound_to_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            record = generate_scenario_splits(directory, split_sizes={"training": 1})[0]
            self.assertEqual(record.scenario_hash, scenario_file_hash(record.path))
            runtime = make_certified_uav_env(record.path, timing_mode="functional")
            runtime.reset(seed=record.seed)
            epoch = runtime.action_context()["certificate_epoch"]
            item = transition(epoch=epoch)
            object.__setattr__(item, "scenario_id", record.scenario_id)
            object.__setattr__(item, "scenario_hash", record.scenario_hash)
            object.__setattr__(item, "certificate_manifest_hash", epoch)
            replay = GeneratorReplayBuffer(4, "group", seed=0)
            self.assertTrue(replay.add(item))
            with self.assertRaises(ValueError):
                object.__setattr__(item, "certificate_manifest_hash", "wrong-manifest")
                replay.add(item)

    def test_out_of_contract_disturbance_fails_certificate_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            record = generate_scenario_splits(directory, split_sizes={"training": 1})[0]
            path = Path(record.path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["mission"]["synthetic_disturbance_fraction"] = 1.1
            path.write_text(json.dumps(payload), encoding="utf-8")
            runtime = make_certified_uav_env(path, timing_mode="functional")
            runtime.reset(seed=0)
            self.assertEqual(runtime.mission_provider.validation_report()["mission_certificate_gate"], "blocked-by-mission-certificate")
            self.assertFalse(runtime.action_context()["generator_available"])

    def test_in_contract_tracking_disturbance_is_applied_within_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            record = generate_scenario_splits(directory, split_sizes={"training": 1})[0]
            path = Path(record.path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["mission"]["synthetic_disturbance_fraction"] = 1.0
            path.write_text(json.dumps(payload), encoding="utf-8")
            runtime = make_certified_uav_env(path, timing_mode="functional")
            observation, _ = runtime.reset(seed=0)
            _, _, _, _, info = runtime.step(StatelessGeneratorPolicy("center_only", 0).select_u(observation))
            trace = info["telemetry"].action_trace
            residual = trace.measured - trace.published
            self.assertTrue(np.any(np.abs(residual) > 0.0))
            self.assertTrue(np.all(np.abs(residual) <= runtime.config.tracking_error_bound + 1e-12))


if __name__ == "__main__":
    unittest.main()
