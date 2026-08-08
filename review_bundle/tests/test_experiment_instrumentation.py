from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from cert_runtime.experiment_metrics import (
    PersistentMetricAccumulator,
    episode_record,
    learning_curve_steps_monotonic,
    metric_snapshot_delta,
    write_jsonl,
)
from scripts.evaluate_persistent_generator_sac import evaluation_episode_seed


def snapshot(
    *,
    tasks: int = 0,
    accepted: int = 0,
    backup: int = 0,
    collisions: int = 0,
    depleted: int = 0,
    uncertified: int = 0,
    margin: float = 5.0,
    task_steps: tuple[int, ...] = (),
) -> dict[str, object]:
    return {
        "tasks_completed": tasks,
        "voluntary_station_arrivals": 0,
        "backup_recovery_count": backup,
        "charging_visits": 0,
        "charging_steps": 0,
        "energy_charged": 0.0,
        "departure_attempts": 0,
        "departure_rejection_count": 0,
        "generator_accepted_steps": accepted,
        "no_generator_steps": 0,
        "energy_consumed": float(accepted),
        "collision_count": collisions,
        "energy_depletion_count": depleted,
        "uncertified_publication_count": uncertified,
        "invalid_kappa_fallback_count": 0,
        "task_completion_steps": list(task_steps),
        "energy_margin_at_backup": [],
        "energy_margin_at_station_approach": [],
        "energy_on_station_arrival": [],
        "energy_on_departure": [],
        "charge_durations": [],
        "minimum_energy_margin": margin,
    }


def observe_episode(
    run: PersistentMetricAccumulator,
    values: list[dict[str, object]],
) -> PersistentMetricAccumulator:
    episode = PersistentMetricAccumulator()
    previous = None
    for index, current in enumerate(values):
        delta = metric_snapshot_delta(previous, current)
        info = {"goal_progress": 0.1 * (index + 1)}
        run.observe(1.0, info, delta)
        episode.observe(1.0, info, delta)
        previous = current
    return episode


class ExperimentInstrumentationTests(unittest.TestCase):
    def test_training_metrics_aggregate_across_episode_resets(self) -> None:
        run = PersistentMetricAccumulator()
        observe_episode(run, [snapshot(accepted=1), snapshot(tasks=1, accepted=2, task_steps=(2,))])
        observe_episode(run, [snapshot(accepted=1), snapshot(tasks=2, accepted=2, task_steps=(1, 2))])
        result = run.summary()
        self.assertEqual(result["total_steps"], 4)
        self.assertEqual(result["tasks_completed"], 3)
        self.assertEqual(result["generator_accepted_steps"], 4)

    def test_training_summary_not_equal_only_last_episode(self) -> None:
        run = PersistentMetricAccumulator()
        observe_episode(run, [snapshot(tasks=2, accepted=1, task_steps=(1, 1))])
        last = observe_episode(run, [snapshot(tasks=1, accepted=1, task_steps=(1,))])
        self.assertEqual(run.summary()["tasks_completed"], 3)
        self.assertEqual(last.summary()["tasks_completed"], 1)

    def test_episode_records_preserved(self) -> None:
        accumulator = PersistentMetricAccumulator()
        observe_episode(accumulator, [snapshot(tasks=1, accepted=1, task_steps=(1,))])
        record = episode_record(
            0,
            10,
            {"sampled_start": np.array((1.0, 2.0, 1.0)), "sampled_goal": np.array((3.0, 2.0, 1.0))},
            accumulator,
            [np.array((3.0, 2.0, 1.0))],
            terminated=False,
            truncated=True,
            partial=False,
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.jsonl"
            write_jsonl(path, [record])
            restored = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(restored["episode_seed"], 10)
        self.assertEqual(restored["tasks_completed"], 1)

    def test_learning_curve_records_monotonic_steps(self) -> None:
        self.assertTrue(learning_curve_steps_monotonic([{"step": 250}, {"step": 500}, {"step": 1000}]))
        self.assertFalse(learning_curve_steps_monotonic([{"step": 500}, {"step": 500}]))

    def test_evaluation_uses_distinct_episode_seeds(self) -> None:
        self.assertEqual([evaluation_episode_seed(100, index) for index in range(3)], [100, 101, 102])

    def test_evaluation_aggregates_all_episodes(self) -> None:
        run = PersistentMetricAccumulator()
        observe_episode(run, [snapshot(tasks=1, accepted=1, task_steps=(1,))])
        observe_episode(run, [snapshot(tasks=2, accepted=1, task_steps=(1, 1))])
        self.assertEqual(run.summary()["tasks_completed"], 3)

    def test_aggregate_safety_counts_sum_episode_counts(self) -> None:
        run = PersistentMetricAccumulator()
        observe_episode(run, [snapshot(collisions=1, depleted=1, uncertified=1)])
        observe_episode(run, [snapshot(collisions=2, depleted=0, uncertified=3)])
        result = run.summary()
        self.assertEqual(result["collision_count"], 3)
        self.assertEqual(result["energy_depletion_count"], 1)
        self.assertEqual(result["uncertified_publication_count"], 4)

    def test_tasks_per_1000_steps_uses_total_run_steps(self) -> None:
        run = PersistentMetricAccumulator()
        observe_episode(run, [snapshot(tasks=1), snapshot(tasks=2)])
        self.assertEqual(run.summary()["tasks_per_1000_steps"], 1000.0)

    def test_minimum_energy_margin_is_global_minimum(self) -> None:
        run = PersistentMetricAccumulator()
        observe_episode(run, [snapshot(margin=4.0), snapshot(margin=2.0)])
        observe_episode(run, [snapshot(margin=3.0), snapshot(margin=1.5)])
        self.assertEqual(run.summary()["minimum_energy_margin"], 1.5)

    def test_goal_sequences_recorded_for_reproducibility(self) -> None:
        accumulator = PersistentMetricAccumulator()
        observe_episode(accumulator, [snapshot()])
        goals = [np.array((1.1, 2.2, 1.0)), np.array((3.3, 2.4, 1.0))]
        record = episode_record(
            1,
            101,
            {},
            accumulator,
            goals,
            terminated=False,
            truncated=False,
            partial=True,
        )
        self.assertEqual(record["goal_sequence"], [goal.tolist() for goal in goals])


if __name__ == "__main__":
    unittest.main()
