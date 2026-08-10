from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cert_runtime.synthetic import build_synthetic_closure_fixture


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one synthetic fixed-corridor proof closure")
    parser.add_argument("--output", type=Path, default=None, help="optional proof-manifest JSON path")
    arguments = parser.parse_args()
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
    payload = asdict(result)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({
        "closed": result.closed,
        "status": result.status,
        "manifest_hash": result.manifest.manifest_hash if result.manifest else None,
        "manifest_entries": len(result.manifest.entries) if result.manifest else 0,
        "dependency_edges": len(result.manifest.dependency_edges) if result.manifest else 0,
        "failure_witness": asdict(result.failure_witness) if result.failure_witness else None,
    }, sort_keys=True))
    return 0 if result.closed else 1


if __name__ == "__main__":
    raise SystemExit(main())
