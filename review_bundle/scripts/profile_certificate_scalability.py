from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from envs.certified_uav import make_certified_uav_env
from envs.certified_uav.mission_certificate import _MANIFEST_CACHE
from experiments.metrics import write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", default=["mission_open", "mission_obstacle", "mission_narrow", "mission_energy_tight"])
    parser.add_argument("--lookup-repetitions", type=int, default=1000)
    parser.add_argument("--output", default="artifacts/paper/certificate_scalability.csv")
    args = parser.parse_args()
    rows = []
    for scenario in args.scenarios:
        _MANIFEST_CACHE.clear()
        started = perf_counter()
        runtime = make_certified_uav_env(f"{scenario}.json", timing_mode="functional")
        runtime.reset(seed=0)
        build_time = perf_counter() - started
        manifest = runtime.mission_provider.manifest
        serialized = pickle.dumps(manifest, protocol=pickle.HIGHEST_PROTOCOL)
        waypoints = np.asarray(runtime.mission_provider.return_waypoints)
        corridor_length = float(np.linalg.norm(np.diff(waypoints, axis=0), axis=1).sum())
        state = runtime._certificate_state()
        lookup_started = perf_counter()
        for _ in range(args.lookup_repetitions):
            runtime.mission_provider._locate_root(state)
        lookup_elapsed = perf_counter() - lookup_started
        rows.append({
            "scenario": scenario,
            "recovery_cells": len(manifest.cells),
            "task_roots": len(runtime.mission_provider.root_cells),
            "corridor_length_m": corridor_length,
            "cells_per_meter": len(manifest.cells) / max(corridor_length, 1e-12),
            "manifest_pickle_bytes": len(serialized),
            "manifest_megabytes": len(serialized) / (1024 * 1024),
            "offline_construction_seconds": build_time,
            "root_lookup_mean_seconds": lookup_elapsed / args.lookup_repetitions,
            "lookup_repetitions": args.lookup_repetitions,
            "index_semantics": "linear root scan followed by exact ellipsoid containment; index is not certificate evidence",
            "evidence_scope": "desktop Python profiling, not WCET",
        })
        print(json.dumps(rows[-1], sort_keys=True), flush=True)
    write_csv(Path(args.output), rows)


if __name__ == "__main__":
    main()
