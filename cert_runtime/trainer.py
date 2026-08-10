from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from json import dumps
from typing import Sequence

from .runtime import ReplayRecord
from .state import CertificateStateSnapshot

try:
    import torch
    from torch import Tensor

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    Tensor = object
    TORCH_AVAILABLE = False


@dataclass(frozen=True)
class CertificateEpoch:
    epoch_id: str
    certificate_version: tuple[int, int, int]
    bound_versions: tuple[tuple[str, str], ...]
    geometry_digest: str
    corridor_digest: str

    @classmethod
    def from_snapshot(cls, snapshot: CertificateStateSnapshot) -> "CertificateEpoch":
        payload = (
            snapshot.certificate_version,
            snapshot.bound_versions,
            snapshot.local_geometry_digest,
            snapshot.return_corridor_digest,
        )
        epoch_id = sha256(dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
        return cls(
            epoch_id,
            snapshot.certificate_version,
            snapshot.bound_versions,
            snapshot.local_geometry_digest,
            snapshot.return_corridor_digest,
        )

    def accepts(self, record: ReplayRecord) -> bool:
        return (
            record.certificate_version == self.certificate_version
            and record.bound_versions == self.bound_versions
            and record.certificate_state.local_geometry_digest == self.geometry_digest
            and record.certificate_state.return_corridor_digest == self.corridor_digest
        )


@dataclass(frozen=True)
class BranchedActorLoss:
    loss: object
    accepted_count: int
    fallback_count: int
    semantics: str = "T12A accepted branch only; fallback atom excluded from generator entropy"


class GeneratorSACTrainer:
    """Minimal epoch-frozen Generator-SAC interface; no environment loop."""

    def __init__(self, actor, critic, alpha: float) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch is required for GeneratorSACTrainer")
        if alpha < 0.0:
            raise ValueError("entropy temperature must be nonnegative")
        self.actor = actor
        self.critic = critic
        self.alpha = alpha
        self.epoch: CertificateEpoch | None = None

    def begin_epoch(self, snapshot: CertificateStateSnapshot) -> CertificateEpoch:
        self.epoch = CertificateEpoch.from_snapshot(snapshot)
        return self.epoch

    def validate_replay(self, records: Sequence[ReplayRecord]) -> None:
        if self.epoch is None:
            raise RuntimeError("certificate epoch must be frozen before optimization")
        if not records or any(not self.epoch.accepts(record) for record in records):
            raise ValueError("replay record does not belong to the frozen certificate epoch")

    def critic_loss(
        self,
        state_features: Tensor,
        records: Sequence[ReplayRecord],
        targets: Tensor,
    ) -> Tensor:
        self.validate_replay(records)
        executed_actions = torch.as_tensor(
            [record.executed_action for record in records],
            dtype=state_features.dtype,
            device=state_features.device,
        )
        predictions = self.critic(state_features, executed_actions).reshape_as(targets)
        return torch.nn.functional.mse_loss(predictions, targets)

    def actor_loss(
        self,
        observations: Tensor,
        state_features: Tensor,
        records: Sequence[ReplayRecord],
    ) -> BranchedActorLoss:
        self.validate_replay(records)
        accepted_indices = [index for index, record in enumerate(records) if record.accepted]
        fallback_count = len(records) - len(accepted_indices)
        if not accepted_indices:
            zero = sum(parameter.sum() * 0.0 for parameter in self.actor.parameters())
            return BranchedActorLoss(zero, 0, fallback_count)
        losses = []
        for index in accepted_indices:
            record = records[index]
            if record.zonotope_center is None or record.zonotope_generators is None:
                raise ValueError("accepted replay record lacks c,G")
            center = torch.as_tensor(
                record.zonotope_center,
                dtype=observations.dtype,
                device=observations.device,
            ).detach()
            generators = torch.as_tensor(
                record.zonotope_generators,
                dtype=observations.dtype,
                device=observations.device,
            ).detach()
            action, log_density, _ = self.actor.action_and_log_density(
                observations[index], center, generators
            )
            q_value = self.critic(state_features[index], action)
            losses.append(self.alpha * log_density - q_value.squeeze())
        return BranchedActorLoss(torch.stack(losses).mean(), len(accepted_indices), fallback_count)
