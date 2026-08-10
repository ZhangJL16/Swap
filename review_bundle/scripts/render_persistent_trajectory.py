#!/usr/bin/env python3
"""Render persistent UAV trajectory JSONL without touching environment dynamics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cert_runtime.trajectory_visualization import read_trajectory, render_trajectory
from envs.certified_uav.scenario import load_scenario


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--scenario", default="random_persistent_open")
    parser.add_argument("--png", required=True)
    parser.add_argument("--gif", required=True)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    render_trajectory(
        read_trajectory(ROOT / args.trajectory),
        load_scenario(f"{args.scenario}.json"),
        ROOT / args.png,
        ROOT / args.gif,
        frame_stride=args.frame_stride,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
