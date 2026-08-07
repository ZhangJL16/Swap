"""Minimal corridor-conditional certification runtime prototype."""

from .certificates import (
    ProofMetadata,
    RecoveryCellCertificate,
    RecoveryEnergyCertificate,
    StateCellBounds,
    TerminalCondition,
)
from .contracts import RuntimeSensorBoundsContract, WCETContract
from .closure import (
    CalibrationBundle,
    CorridorClosureResult,
    FailureWitness,
    ProofManifest,
    ProofManifestEntry,
    SingleCorridorClosurePipeline,
)
from .corridor import CorridorCell, ReturnCorridor
from .energy import RecoveryEnergySolver
from .envelope import DynamicsBounds, EnergyBounds, SuccessorEnvelope, SuccessorEnvelopeBuilder
from .geometry import CellState, LidarRay, RollingLocalGeometry, SensorBounds
from .interval import Interval
from .recovery import CorridorRecoveryVerifier, FrozenRecoveryPolicy, RecoveryConfig
from .runtime import CertificateReplay, RuntimeCertifier
from .state import CertificateState, CertificateStateSnapshot
from .types import AABB2, Interval3, Zonotope3
from .watchdog import AtomicCommandPublisher, CandidateBundle, SimulatedWatchdog
from .zonotope import CertificateConfig, ZonotopeCertificate, ZonotopeConstructor

__all__ = [
    "AABB2",
    "AtomicCommandPublisher",
    "CandidateBundle",
    "CalibrationBundle",
    "CellState",
    "CertificateConfig",
    "CertificateReplay",
    "CertificateState",
    "CertificateStateSnapshot",
    "CorridorCell",
    "CorridorClosureResult",
    "CorridorRecoveryVerifier",
    "DynamicsBounds",
    "EnergyBounds",
    "FrozenRecoveryPolicy",
    "FailureWitness",
    "Interval",
    "Interval3",
    "LidarRay",
    "RecoveryCellCertificate",
    "RecoveryConfig",
    "RecoveryEnergyCertificate",
    "RecoveryEnergySolver",
    "ReturnCorridor",
    "RollingLocalGeometry",
    "RuntimeCertifier",
    "RuntimeSensorBoundsContract",
    "SensorBounds",
    "SimulatedWatchdog",
    "StateCellBounds",
    "ProofManifest",
    "ProofManifestEntry",
    "ProofMetadata",
    "SingleCorridorClosurePipeline",
    "SuccessorEnvelope",
    "SuccessorEnvelopeBuilder",
    "TerminalCondition",
    "WCETContract",
    "Zonotope3",
    "ZonotopeCertificate",
    "ZonotopeConstructor",
]
