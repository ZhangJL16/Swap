from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from cert_runtime.generator_sac import GeneratorSAC, GeneratorSACConfig
from envs.certified_uav import make_certified_uav_env
from experiments.agents import DirectSACAgent, StatelessGeneratorPolicy
from experiments.metrics import write_csv
from experiments.runner import _evaluate


def _checkpoint(root: Path, family: str, method: str, seed: int) -> Path:
    return root / f"mission_{family}" / method / f"seed_{seed}" / "checkpoint_latest.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-index", default="artifacts/scenario_families/scenario_index.json")
    parser.add_argument("--checkpoint-root", default="artifacts/comparison")
    parser.add_argument("--methods", nargs="+", default=["generator_sac", "center_only", "shield_sac"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", default="artifacts/paper/generalization.csv")
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument("--scenario-ids", nargs="+", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    records = [record for record in json.loads(Path(args.scenario_index).read_text(encoding="utf-8")) if record["split"] == "heldout"]
    if args.families is not None:
        records = [record for record in records if record["family"] in set(args.families)]
    if args.scenario_ids is not None:
        records = [record for record in records if record["scenario_id"] in set(args.scenario_ids)]
    rows = []
    blockers = []
    for record in records:
        for method in args.methods:
            if method not in {"generator_sac", "shield_sac"}:
                continue
            for seed in args.seeds:
                checkpoint = _checkpoint(Path(args.checkpoint_root), record["family"], method, seed)
                if not checkpoint.exists():
                    blockers.append(f"{record['scenario_id']}:{method}:seed-{seed}:missing-checkpoint")
    if blockers and all(method in {"generator_sac", "shield_sac"} for method in args.methods):
        write_csv(Path(args.output), rows)
        gate = {
            "GENERALIZATION_GATE": "BLOCKED",
            "evaluated_rows": 0,
            "blocking_items": blockers,
            "scope": "held-out synthetic empirical evidence; not physical calibration",
        }
        Path(args.output).with_name("generalization_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
        print(json.dumps(gate, indent=2))
        raise SystemExit(2)
    for record in records:
        runtime = make_certified_uav_env(record["path"], timing_mode="functional")
        observation, _ = runtime.reset(seed=record["seed"])
        actual_manifest = runtime.mission_provider.manifest.manifest_hash
        manifest_matches = actual_manifest == record.get("certificate_manifest_hash")
        gate_pass = runtime.mission_provider.gate_pass and manifest_matches
        if not gate_pass:
            blockers.append(f"{record['scenario_id']}:certificate")
            continue
        for method in args.methods:
            for seed in args.seeds:
                checkpoint = None
                if method == "center_only":
                    agent = StatelessGeneratorPolicy("center_only", seed)
                elif method == "generator_sac":
                    checkpoint = _checkpoint(Path(args.checkpoint_root), record["family"], method, seed)
                    if not checkpoint.exists():
                        continue
                    agent = GeneratorSAC(observation.size, GeneratorSACConfig(batch_size=64, hidden_dim=128), seed=seed, device=args.device)
                    agent.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=False))
                elif method == "shield_sac":
                    checkpoint = _checkpoint(Path(args.checkpoint_root), record["family"], method, seed)
                    if not checkpoint.exists():
                        continue
                    agent = DirectSACAgent(observation.size, runtime.config.a_max, seed=seed, device=args.device)
                    agent.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=False))
                else:
                    raise ValueError(f"unsupported held-out method {method}")
                episodes = _evaluate(method, agent, record["path"], seed * 1000, args.episodes, "task_oriented", "functional")
                rows.append({
                    "scenario_id": record["scenario_id"],
                    "scenario_family": record["family"],
                    "scenario_hash": record["scenario_hash"],
                    "geometry_hash": record["geometry_hash"],
                    "certificate_manifest_hash": actual_manifest,
                    "method": method,
                    "seed": seed,
                    "episodes": len(episodes),
                    "task_success": float(np.mean([item["task_success"] for item in episodes])),
                    "return_success": float(np.mean([item["return_success"] for item in episodes])),
                    "collision": float(np.mean([item["collision"] for item in episodes])),
                    "outbound_intervention": float(np.mean([item["outbound_intervention_rate"] for item in episodes])),
                    "path_length": float(np.mean([item["total_path_length"] for item in episodes])),
                    "energy_consumed": float(np.mean([item["total_energy_consumed"] for item in episodes])),
                    "uncertified_task_publication_count": int(sum(item["uncertified_task_publication_count"] for item in episodes)),
                    "invalid_kappa_fallback_count": int(sum(item["invalid_kappa_fallback_count"] for item in episodes)),
                    "checkpoint": "not_applicable" if checkpoint is None else str(checkpoint),
                    "evidence_scope": "held-out synthetic empirical evaluation",
                })
                write_csv(Path(args.output), rows)
                print(json.dumps(rows[-1]), flush=True)
    write_csv(Path(args.output), rows)
    gate = {
        "GENERALIZATION_GATE": "PASS" if rows and not blockers else "BLOCKED",
        "evaluated_rows": len(rows),
        "blocking_items": blockers,
        "scope": "held-out synthetic empirical evidence; not physical calibration",
    }
    gate_path = Path(args.output).with_name("generalization_gate.json")
    gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2))
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
