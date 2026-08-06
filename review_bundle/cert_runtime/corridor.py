from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from json import dumps

from .certificates import (
    RecoveryCellCertificate,
    RecoveryEnergyCertificate,
    StateCellBounds,
)
from .geometry import RollingLocalGeometry
from .types import AABB2, Interval3


@dataclass(frozen=True)
class CorridorCell:
    cell_id: int
    region: AABB2
    maximum_speed: float
    geometry_version: int
    state_bounds: StateCellBounds
    valid: bool = True
    recovery_certificate: RecoveryCellCertificate | None = None
    energy_certificate: RecoveryEnergyCertificate | None = None

    @property
    def recovery_energy_upper(self) -> float | None:
        return self.energy_certificate.transit_energy_upper if self.energy_certificate else None


@dataclass(frozen=True)
class CorridorUpdateResult:
    valid: bool
    migration_required: bool
    fallback_required: bool
    removed_cell_ids: tuple[int, ...] = ()
    migration_target: AABB2 | None = None


class ReturnCorridor:
    """Finite station-to-UAV chain of overlapping proof-carrying AABBs."""

    def __init__(self, transfer_radius: float, geometry_margin: float) -> None:
        if transfer_radius < 0.0 or geometry_margin < 0.0:
            raise ValueError("corridor margins must be nonnegative")
        self.transfer_radius = transfer_radius
        self.geometry_margin = geometry_margin
        self.cells: list[CorridorCell] = []
        self.version = 0
        self.certificate_epoch = 0

    @property
    def certificate_version(self) -> tuple[int, int]:
        return (self.version, self.certificate_epoch)

    def certificate_digest(self) -> str:
        payload = {
            "version": self.version,
            "certificate_epoch": self.certificate_epoch,
            "cells": [
                {
                    "id": cell.cell_id,
                    "region": repr(cell.region),
                    "state_bounds": repr(cell.state_bounds),
                    "geometry_version": cell.geometry_version,
                    "valid": cell.valid,
                    "recovery_hash": (
                        cell.recovery_certificate.certificate_hash
                        if cell.recovery_certificate
                        else None
                    ),
                    "energy_hash": (
                        cell.energy_certificate.certificate_hash
                        if cell.energy_certificate
                        else None
                    ),
                }
                for cell in self.cells
            ],
        }
        return sha256(dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def create(self, cells: list[CorridorCell], geometry: RollingLocalGeometry) -> bool:
        if not cells or cells[0].cell_id != 0:
            return False
        if [cell.cell_id for cell in cells] != list(range(len(cells))):
            return False
        if not all(self._cell_geometry_is_valid(cell, geometry) for cell in cells):
            return False
        if not all(self._overlap_is_valid(left, right) for left, right in zip(cells, cells[1:])):
            return False
        self.cells = [
            replace(
                cell,
                geometry_version=geometry.version,
                recovery_certificate=None,
                energy_certificate=None,
            )
            for cell in cells
        ]
        self.version += 1
        self.certificate_epoch += 1
        return True

    def extend(self, cell: CorridorCell, geometry: RollingLocalGeometry) -> bool:
        if not self.cells or cell.cell_id != self.cells[-1].cell_id + 1:
            return False
        if not self._cell_geometry_is_valid(cell, geometry) or not self._overlap_is_valid(self.cells[-1], cell):
            return False
        self.cells.append(
            replace(
                cell,
                geometry_version=geometry.version,
                recovery_certificate=None,
                energy_certificate=None,
            )
        )
        self.version += 1
        self.certificate_epoch += 1
        return True

    def install_recovery_certificates(
        self,
        certificates: dict[int, RecoveryCellCertificate],
    ) -> bool:
        if set(certificates) != {cell.cell_id for cell in self.cells}:
            return False
        for cell in self.cells:
            certificate = certificates[cell.cell_id]
            if (
                certificate.cell_id != cell.cell_id
                or certificate.level != cell.cell_id
                or certificate.corridor_version != self.version
                or certificate.geometry_version != cell.geometry_version
                or certificate.state_bounds != cell.state_bounds
                or certificate.certificate_hash != certificate.expected_hash
            ):
                return False
        self.cells = [
            replace(cell, recovery_certificate=certificates[cell.cell_id], energy_certificate=None)
            for cell in self.cells
        ]
        self.certificate_epoch += 1
        return True

    def install_energy_certificates(
        self,
        certificates: dict[int, RecoveryEnergyCertificate],
    ) -> bool:
        if set(certificates) != {cell.cell_id for cell in self.cells}:
            return False
        for cell in self.cells:
            recovery = cell.recovery_certificate
            energy = certificates[cell.cell_id]
            if (
                recovery is None
                or recovery.certificate_hash != recovery.expected_hash
                or energy.cell_id != cell.cell_id
                or energy.level != cell.cell_id
                or energy.certificate_hash != energy.expected_hash
                or energy.recovery_certificate_hash != recovery.certificate_hash
            ):
                return False
            if energy.corridor_version != self.version:
                return False
        self.cells = [replace(cell, energy_certificate=certificates[cell.cell_id]) for cell in self.cells]
        self.certificate_epoch += 1
        return True

    def invalidate_certificates(self) -> None:
        self.cells = [replace(cell, recovery_certificate=None, energy_certificate=None) for cell in self.cells]
        self.certificate_epoch += 1

    def revalidate(
        self,
        geometry: RollingLocalGeometry,
        current_position: tuple[float, float],
    ) -> CorridorUpdateResult:
        first_invalid = next(
            (index for index, cell in enumerate(self.cells) if not self._cell_geometry_is_valid(cell, geometry)),
            None,
        )
        if first_invalid is None:
            self.cells = [
                replace(
                    cell,
                    geometry_version=geometry.version,
                    recovery_certificate=None,
                    energy_certificate=None,
                )
                for cell in self.cells
            ]
            self.version += 1
            self.certificate_epoch += 1
            return CorridorUpdateResult(True, False, False)
        invalid_suffix = self.cells[first_invalid:]
        valid_prefix = self.cells[:first_invalid]
        if valid_prefix and any(cell.region.contains_point(current_position) for cell in valid_prefix):
            self.cells = [
                replace(
                    cell,
                    geometry_version=geometry.version,
                    recovery_certificate=None,
                    energy_certificate=None,
                )
                for cell in valid_prefix
            ]
            self.version += 1
            self.certificate_epoch += 1
            return CorridorUpdateResult(True, False, False, tuple(cell.cell_id for cell in invalid_suffix))
        target = valid_prefix[-1].region if valid_prefix else None
        self.cells = [
            replace(
                cell,
                valid=False,
                recovery_certificate=None,
                energy_certificate=None,
            )
            if index >= first_invalid
            else replace(
                cell,
                geometry_version=geometry.version,
                recovery_certificate=None,
                energy_certificate=None,
            )
            for index, cell in enumerate(self.cells)
        ]
        self.version += 1
        self.certificate_epoch += 1
        return CorridorUpdateResult(False, target is not None, True, (), target)

    def complete_migration(self, current_position: tuple[float, float]) -> CorridorUpdateResult:
        first_invalid = next((index for index, cell in enumerate(self.cells) if not cell.valid), None)
        if first_invalid is None:
            return CorridorUpdateResult(True, False, False)
        valid_prefix = self.cells[:first_invalid]
        if not valid_prefix or not valid_prefix[-1].region.contains_point(current_position):
            return CorridorUpdateResult(
                False,
                bool(valid_prefix),
                True,
                migration_target=valid_prefix[-1].region if valid_prefix else None,
            )
        removed = tuple(cell.cell_id for cell in self.cells[first_invalid:])
        self.cells = [replace(cell, recovery_certificate=None, energy_certificate=None) for cell in valid_prefix]
        self.version += 1
        self.certificate_epoch += 1
        return CorridorUpdateResult(True, False, False, removed)

    def locate_level(self, position: tuple[float, float]) -> int | None:
        containing = [cell.cell_id for cell in self.cells if cell.valid and cell.region.contains_point(position)]
        return max(containing) if containing else None

    def lower_level_cell(self, level: int) -> CorridorCell | None:
        if level <= 0:
            return None
        return next((cell for cell in self.cells if cell.cell_id == level - 1 and cell.valid), None)

    def containing_cell_for_envelope(
        self,
        position: Interval3,
        velocity: Interval3,
        energy_low: float,
        energy_reserve: float,
        terminal_energy: float,
        tolerance: float,
        geometry_version: int,
        kappa_parameter_version: str,
        dynamics_bound_version: str,
        energy_bound_version: str,
        timestamp: float,
    ) -> CorridorCell | None:
        position_box = position.horizontal_box()
        velocity_abs = velocity.max_abs()
        for cell in reversed(self.cells):
            recovery = cell.recovery_certificate
            energy = cell.energy_certificate
            if not cell.valid or recovery is None or energy is None:
                continue
            if not recovery.is_valid(
                geometry_version,
                self.version,
                kappa_parameter_version,
                dynamics_bound_version,
                energy_bound_version,
                timestamp,
            ):
                continue
            if not energy.is_valid(
                recovery.certificate_hash,
                energy_bound_version,
                self.version,
                timestamp,
            ):
                continue
            if not cell.region.contains_box(position_box, tolerance):
                continue
            if max(velocity_abs[0], velocity_abs[1]) > cell.maximum_speed + tolerance:
                continue
            if energy_low + tolerance < energy.transit_energy_upper + energy_reserve + terminal_energy:
                continue
            return cell
        return None

    def _cell_geometry_is_valid(self, cell: CorridorCell, geometry: RollingLocalGeometry) -> bool:
        position_box = cell.state_bounds.position.horizontal_box()
        velocity_abs = cell.state_bounds.velocity.max_abs()
        return (
            cell.valid
            and cell.region.contains_box(position_box)
            and max(velocity_abs[0], velocity_abs[1]) <= cell.maximum_speed
            and geometry.box_is_verified_free(cell.region, self.geometry_margin)
        )

    def _overlap_is_valid(self, left: CorridorCell, right: CorridorCell) -> bool:
        overlap = left.region.intersection(right.region)
        required_width = 2.0 * self.transfer_radius
        return overlap is not None and overlap.width >= required_width and overlap.height >= required_width
