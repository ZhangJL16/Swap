from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


class ExecutionAuthority(str, Enum):
    RL_GENERATOR = "RL_GENERATOR"
    KAPPA_BACKUP = "KAPPA_BACKUP"
    CHARGER_CONSTRAINED = "CHARGER_CONSTRAINED"
    FAIL_CLOSED = "FAIL_CLOSED"


@dataclass(frozen=True, slots=True)
class PersistentAuthorityInput:
    persistent_mode: str
    energy_margin: float
    backup_switch_margin: float
    persistent_certificate_valid: bool
    certificate_valid: bool
    kappa_valid: bool
    generator_available: bool
    recoverable_set_member: bool
    recoverability_action_verified: bool
    policy_authority_pass: bool
    charging_state: bool
    departure_allowed: bool
    charging_support_verified: bool
    station_hold_valid: bool


@dataclass(frozen=True, slots=True)
class PersistentAuthorityDecision:
    authority: ExecutionAuthority
    reason: str
    generator_executable: bool
    kappa_required: bool
    departure_allowed: bool
    charging_restriction: bool
    station_hold_required: bool


class PersistentExecutionAuthority:
    """Pure execution classification shared by runtime and persistent SAC."""

    @staticmethod
    def evaluate(inputs: PersistentAuthorityInput) -> PersistentAuthorityDecision:
        if not inputs.kappa_valid:
            return PersistentAuthorityDecision(
                ExecutionAuthority.FAIL_CLOSED,
                "KAPPA_CERTIFICATE_INVALID",
                False,
                False,
                False,
                inputs.charging_state and not inputs.departure_allowed,
                False,
            )
        if inputs.persistent_mode == "BACKUP_RECOVERY":
            return PersistentAuthorityDecision(
                ExecutionAuthority.KAPPA_BACKUP,
                "BACKUP_RECOVERY_CONTINUATION",
                False,
                True,
                False,
                False,
                False,
            )
        state_checks = (
            (inputs.persistent_certificate_valid, "PERSISTENT_CERTIFICATE_GATE_FAILED"),
            (inputs.certificate_valid, "RECOVERY_CERTIFICATE_INVALID"),
            (inputs.recoverable_set_member, "RECOVERABLE_SET_CERTIFICATE_INVALID"),
            (isfinite(inputs.energy_margin), "ENERGY_MARGIN_NONFINITE"),
            (inputs.energy_margin > inputs.backup_switch_margin, "ENERGY_MARGIN_BACKUP_SWITCH"),
        )
        for valid, reason in state_checks:
            if not valid:
                return PersistentAuthorityDecision(
                    ExecutionAuthority.KAPPA_BACKUP,
                    reason,
                    False,
                    True,
                    False,
                    inputs.charging_state and not inputs.departure_allowed,
                    False,
                )
        if inputs.charging_state and not inputs.departure_allowed:
            if (
                inputs.generator_available
                and inputs.recoverability_action_verified
                and inputs.policy_authority_pass
                and inputs.charging_support_verified
            ):
                return PersistentAuthorityDecision(
                    ExecutionAuthority.CHARGER_CONSTRAINED,
                    "VERIFIED_CHARGING_SUPPORT",
                    True,
                    False,
                    False,
                    True,
                    False,
                )
            if inputs.station_hold_valid:
                return PersistentAuthorityDecision(
                    ExecutionAuthority.CHARGER_CONSTRAINED,
                    "CHARGING_SUPPORT_UNAVAILABLE_USE_HOLD",
                    False,
                    False,
                    False,
                    True,
                    True,
                )
            return PersistentAuthorityDecision(
                ExecutionAuthority.FAIL_CLOSED,
                "CHARGING_SUPPORT_AND_HOLD_INVALID",
                False,
                False,
                False,
                True,
                False,
            )
        generator_checks = (
            (inputs.generator_available, "NO_GENERATOR_SET"),
            (inputs.recoverability_action_verified, "GENERATOR_NOT_CONTAINED_IN_A_REC"),
            (inputs.policy_authority_pass, "POLICY_AUTHORITY_GATE_FAILED"),
        )
        for valid, reason in generator_checks:
            if not valid:
                return PersistentAuthorityDecision(
                    ExecutionAuthority.KAPPA_BACKUP,
                    reason,
                    False,
                    True,
                    False,
                    False,
                    False,
                )
        return PersistentAuthorityDecision(
            ExecutionAuthority.RL_GENERATOR,
            "VERIFIED_RL_GENERATOR_AUTHORITY",
            True,
            False,
            inputs.departure_allowed if inputs.charging_state else True,
            False,
            False,
        )
