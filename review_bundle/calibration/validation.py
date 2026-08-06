from __future__ import annotations

from collections import Counter
from typing import Iterable

from .schema import DataSplit, OutlierPolicy, RawCalibrationRecord


class CalibrationRegistry:
    """Rejects silent reuse of a contract version with different evidence."""

    def __init__(self) -> None:
        self._hashes: dict[tuple[str, str], str] = {}

    def register(self, contract_kind: str, version: str, contract_hash: str) -> None:
        if not contract_kind or not version or not contract_hash:
            raise ValueError("calibration registry requires kind, version, and hash")
        key = (contract_kind, version)
        previous = self._hashes.get(key)
        if previous is not None and previous != contract_hash:
            raise ValueError("calibration version was reused with different evidence")
        self._hashes[key] = contract_hash

    def fingerprint(self, contract_kind: str, version: str) -> str | None:
        return self._hashes.get((contract_kind, version))


def validate_split_separation(records: Iterable[RawCalibrationRecord]) -> dict[DataSplit, int]:
    records_tuple = tuple(records)
    identifiers = [record.record_id for record in records_tuple]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("record identifiers must be unique across data splits")
    counts = Counter(record.split for record in records_tuple)
    if counts[DataSplit.CALIBRATION] == 0 or counts[DataSplit.VALIDATION] == 0:
        raise ValueError("independent calibration and validation splits are required")
    return {split: counts[split] for split in DataSplit}


def retained_records(
    records: Iterable[RawCalibrationRecord],
    policy: OutlierPolicy,
) -> tuple[RawCalibrationRecord, ...]:
    records_tuple = tuple(records)
    if policy == OutlierPolicy.WINSORIZE_EXPLORATORY:
        raise ValueError("winsorized data cannot produce a strict certificate bound")
    if policy == OutlierPolicy.RETAIN_ALL:
        if any(not record.valid_measurement for record in records_tuple):
            raise ValueError("retain-all policy rejects datasets containing invalid measurements")
        return records_tuple
    return tuple(record for record in records_tuple if record.valid_measurement)


def validation_exceedances(records: Iterable[RawCalibrationRecord], bound: float) -> int:
    return sum(record.absolute_residual > bound for record in records)
