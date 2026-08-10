from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .zonotope import ZonotopeCertificate


@dataclass(frozen=True)
class GeneratorDeploymentReport:
    state_count: int
    generator_enabled_fraction: float
    no_generator_set_fraction: float
    fallback_fraction: float
    mean_volume: float
    minimum_sigma: float | None
    maximum_condition_number: float | None
    mean_construction_time_seconds: float
    mean_verifier_calls: float
    mean_bisection_steps: float
    maximum_verifier_calls: int
    predicate_failure_distribution: tuple[tuple[str, int], ...]


class GeneratorStatistics:
    def __init__(self, maximum_verifier_calls: int) -> None:
        self.maximum_verifier_calls = maximum_verifier_calls
        self.certificates: list[ZonotopeCertificate] = []

    def observe(self, certificate: ZonotopeCertificate) -> None:
        self.certificates.append(certificate)

    def report(self) -> GeneratorDeploymentReport:
        count = len(self.certificates)
        verified = [certificate for certificate in self.certificates if certificate.verified and certificate.zonotope]
        failures = Counter(certificate.reason for certificate in self.certificates if not certificate.verified)
        volumes = [8.0 * abs(certificate.zonotope.determinant) for certificate in verified]
        sigmas = [certificate.zonotope.sigma_min_lower_bound for certificate in verified]
        conditions = [certificate.zonotope.condition_number_upper_bound for certificate in verified]
        return GeneratorDeploymentReport(
            count,
            len(verified) / count if count else 0.0,
            failures["NO_GENERATOR_SET"] / count if count else 0.0,
            (count - len(verified)) / count if count else 0.0,
            sum(volumes) / len(volumes) if volumes else 0.0,
            min(sigmas) if sigmas else None,
            max(conditions) if conditions else None,
            sum(certificate.elapsed_seconds for certificate in self.certificates) / count if count else 0.0,
            sum(certificate.verifier_calls for certificate in self.certificates) / count if count else 0.0,
            sum(certificate.bisection_steps for certificate in self.certificates) / count if count else 0.0,
            self.maximum_verifier_calls,
            tuple(sorted(failures.items())),
        )
