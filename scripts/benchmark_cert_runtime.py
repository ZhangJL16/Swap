from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cert_runtime.contracts import WCETContract
from cert_runtime.synthetic import build_synthetic_closure_fixture
from cert_runtime.wcet import WCETBenchmarkHarness


def main() -> int:
    parser = argparse.ArgumentParser(description="Desktop profiling only; not hard-WCET evidence")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    arguments = parser.parse_args()
    fixture = build_synthetic_closure_fixture()

    def geometry_stage(size: int):
        result = None
        for _ in range(size):
            result = fixture.geometry.certificate_digest()
        return result

    def set_stage(size: int):
        result = None
        for _ in range(size):
            result = fixture.closure.close(
                fixture.state, fixture.geometry, fixture.corridor, fixture.cells,
                fixture.operating_point, fixture.device_version, fixture.firmware_version,
                fixture.timestamp, allow_synthetic=True,
            )
        return result

    contract = WCETContract(control_period_seconds=0.5, margin_seconds=0.05)
    report = WCETBenchmarkHarness(contract).run(
        {"geometry_digest": geometry_stage, "corridor_set": set_stage},
        (1, 2),
        warmup_runs=arguments.warmup,
        measured_runs=arguments.runs,
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
