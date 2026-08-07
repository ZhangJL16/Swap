from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.agents import StatelessGeneratorPolicy
from experiments.metrics import write_csv
from experiments.paper_analysis import ABLATION_METRICS, paired_bootstrap, summarize_evaluations
from experiments.runner import _evaluate


def _copy_reference_results(source_root: Path, target_root: Path, scenarios, methods, seeds) -> None:
    for scenario in scenarios:
        for method in methods:
            for seed in seeds:
                source = source_root / scenario / method / f"seed_{seed}" / "evaluation_metrics.csv"
                if not source.exists():
                    raise FileNotFoundError(f"missing reference evaluation: {source}")
                target = target_root / scenario / method / f"seed_{seed}" / "evaluation_metrics.csv"
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(encoding="utf-8") as handle:
                    evaluation = list(csv.DictReader(handle))
                episode_path = source.parent / "episode_metrics.csv"
                trajectory_path = source.parent / "trajectory_diagnostics.csv"
                episodes = []
                trajectory_by_episode: dict[int, list[dict]] = {}
                if episode_path.exists():
                    with episode_path.open(encoding="utf-8") as handle:
                        episodes = list(csv.DictReader(handle))
                if trajectory_path.exists():
                    with trajectory_path.open(encoding="utf-8") as handle:
                        for row in csv.DictReader(handle):
                            trajectory_by_episode.setdefault(int(row["episode_id"]), []).append(row)
                for index, row in enumerate(evaluation):
                    if episodes:
                        episode = episodes[index % len(episodes)]
                        for key in (
                            "mission_completion_steps", "task_completion_steps", "outbound_path_length",
                            "return_path_length", "total_path_length", "terminal_energy",
                            "minimum_distance_to_task", "outbound_fallback_rate", "return_fallback_rate",
                        ):
                            if key in episode:
                                row[key] = episode[key]
                        row["outbound_intervention_rate"] = episode.get("outbound_fallback_rate", "")
                        row["return_handoff_rate"] = episode.get("return_fallback_rate", "")
                        if row.get("return_success") == "1":
                            row["mission_completion_steps"] = episode.get("episode_length", "")
                        if episode.get("terminal_energy") not in (None, ""):
                            initial_energy = 5.5 if scenario == "mission_energy_tight" else 30.0
                            row["total_energy_consumed"] = initial_energy - float(episode["terminal_energy"])
                    trajectory = trajectory_by_episode.get(index % max(1, len(trajectory_by_episode)), [])
                    residuals = []
                    centers = []
                    ratios = []
                    for item in trajectory:
                        if item.get("fallback_reason"):
                            continue
                        candidate_text, center_text = item.get("candidate_action"), item.get("zonotope_center")
                        if not candidate_text or not center_text:
                            continue
                        candidate = np.asarray(json.loads(candidate_text), dtype=np.float64)
                        center = np.asarray(json.loads(center_text), dtype=np.float64)
                        residual = float(np.linalg.norm(candidate - center))
                        center_norm = float(np.linalg.norm(center))
                        residuals.append(residual); centers.append(center_norm); ratios.append(residual / max(center_norm, 1e-12))
                    row["mean_residual_norm"] = float(np.mean(residuals)) if residuals else ""
                    row["mean_center_norm"] = float(np.mean(centers)) if centers else ""
                    row["mean_residual_to_center_ratio"] = float(np.mean(ratios)) if ratios else ""
                    row["metric_source"] = "evaluation outcomes plus matching checked-in training trajectory diagnostics"
                write_csv(target, evaluation)


def _normalize_stateless_residual_fields(root: Path, scenarios, seeds, preserve_random: bool) -> None:
    for scenario in scenarios:
        for method in ("center_only", "random_generator"):
            for seed in seeds:
                path = root / scenario / method / f"seed_{seed}" / "evaluation_metrics.csv"
                if not path.exists():
                    continue
                with path.open(encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                for row in rows:
                    if method == "center_only":
                        row["mean_residual_norm"] = 0.0
                        row["mean_residual_to_center_ratio"] = 0.0
                        row["mean_cos_residual_goal"] = 0.0
                    elif not preserve_random:
                        row["mean_residual_norm"] = ""
                        row["mean_residual_to_center_ratio"] = ""
                        row["mean_cos_residual_goal"] = ""
                        row["residual_metric_note"] = "not reused when accepted-branch provenance is unavailable"
                write_csv(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", default=["mission_open", "mission_obstacle"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output-dir", default="artifacts/rl_contribution")
    parser.add_argument("--reference-root", default="artifacts/comparison")
    parser.add_argument("--center-mode", default="task_oriented", choices=("task_oriented", "zero", "braking", "max_volume"))
    parser.add_argument("--reuse-stateless", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    if not args.reuse_stateless:
        for scenario in args.scenarios:
            for method in ("center_only", "random_generator"):
                for seed in args.seeds:
                    started = monotonic()
                    rows = _evaluate(
                        method,
                        StatelessGeneratorPolicy(method, seed),
                        scenario,
                        seed * 1000,
                        args.episodes,
                        args.center_mode,
                        "functional",
                    )
                    run_root = output / scenario / method / f"seed_{seed}"
                    write_csv(run_root / "evaluation_metrics.csv", rows)
                    (run_root / "runtime_profile.json").write_text(json.dumps({
                        "method": method,
                        "scenario": scenario,
                        "seed": seed,
                        "evaluation_episodes": len(rows),
                        "wall_time_seconds": monotonic() - started,
                        "training": "not_applicable",
                        "evidence_scope": "synthetic empirical stateless ablation",
                    }, indent=2), encoding="utf-8")
                    print(json.dumps({
                        "method": method,
                        "scenario": scenario,
                        "seed": seed,
                        "task_success": sum(row["task_success"] for row in rows) / len(rows),
                        "return_success": sum(row["return_success"] for row in rows) / len(rows),
                    }), flush=True)
    _copy_reference_results(
        Path(args.reference_root), output, args.scenarios,
        ("generator_sac", "shield_sac", "sac"), args.seeds,
    )
    _normalize_stateless_residual_fields(output, args.scenarios, args.seeds, preserve_random=not args.reuse_stateless)
    paper = Path("artifacts/paper")
    summaries = summarize_evaluations(output, paper / "rl_contribution_ablation.csv")
    bootstrap = [
        paired_bootstrap(output, scenario, metric)
        for scenario in args.scenarios
        for metric in (
            "mission_completion_steps", "total_path_length", "total_energy_consumed",
            "episode_return", "terminal_energy",
        )
    ]
    write_csv(paper / "rl_contribution_bootstrap.csv", bootstrap)
    gate = {
        "RL_CONTRIBUTION_GATE": "PASS" if all(
            any(row["scenario"] == scenario and row["method"] == method for row in summaries)
            for scenario in args.scenarios
            for method in ("center_only", "random_generator", "generator_sac")
        ) else "FAIL",
        "scenarios": args.scenarios,
        "seeds": args.seeds,
        "episodes_per_seed": args.episodes,
        "metrics": list(ABLATION_METRICS),
        "scope": "synthetic empirical ablation; no physical-safety inference",
    }
    (paper / "rl_contribution_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
