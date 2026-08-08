from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np

from cert_runtime.persistent_authority import (
    ExecutionAuthority,
    PersistentAuthorityDecision,
    PersistentAuthorityInput,
    PersistentExecutionAuthority,
)
from cert_runtime.energy_management import (
    EnergyDecision,
    EnergyManagementPolicy,
    EnergyManagementTransition,
)

from .charging import ChargingConfig, ChargingDynamics, DepartureGateResult, verify_departure_energy
from .persistent_certificate import PersistentGoalCertificateProvider
from .persistent_task import CertifiedGoalNetwork, PersistentGoalWrapper, PersistentMissionMode
from .runtime_wrapper import CertifiedRuntimeWrapper


@dataclass(slots=True)
class PersistentMetrics:
    tasks_completed: int = 0
    task_interruption_count: int = 0
    energy_consumed: float = 0.0
    energy_charged: float = 0.0
    charging_visits: int = 0
    charging_steps: int = 0
    voluntary_return_count: int = 0
    forced_return_count: int = 0
    energy_management_override_count: int = 0
    departure_rejection_count: int = 0
    serving_steps: int = 0
    returning_steps: int = 0
    collision_count: int = 0
    energy_depletion_count: int = 0
    uncertified_publication_count: int = 0
    invalid_kappa_fallback_count: int = 0
    total_steps: int = 0
    energy_on_return_request: list[float] = field(default_factory=list)
    energy_on_station_arrival: list[float] = field(default_factory=list)
    energy_on_departure: list[float] = field(default_factory=list)
    charge_durations: list[int] = field(default_factory=list)
    task_completion_steps: list[int] = field(default_factory=list)
    voluntary_station_arrivals: int = 0
    backup_recovery_count: int = 0
    departure_attempts: int = 0
    generator_accepted_steps: int = 0
    no_generator_steps: int = 0
    minimum_energy_margin: float = float("inf")
    energy_margin_at_station_approach: list[float] = field(default_factory=list)
    energy_margin_at_backup: list[float] = field(default_factory=list)


@dataclass(slots=True)
class _DecisionAccumulator:
    observation: np.ndarray
    requested: int
    executed: int
    cumulative_reward: float = 0.0
    duration_steps: int = 0


class LegacyEnergyManagementRuntimeWrapper(gym.Env[np.ndarray, np.ndarray]):
    """Ablation-only two-policy runtime retained for compatibility."""

    metadata = {"render_modes": []}
    energy_observation_dim = 13

    def __init__(
        self,
        runtime: CertifiedRuntimeWrapper,
        network: CertifiedGoalNetwork,
        energy_management_policy: EnergyManagementPolicy,
        charging_config: ChargingConfig | None = None,
        deterministic_energy_management: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(runtime.task_env, PersistentGoalWrapper):
            raise TypeError("persistent runtime requires PersistentGoalWrapper")
        if runtime.config.terminate_on_terminal:
            raise ValueError("persistent plant must set terminate_on_terminal=False")
        self.runtime = runtime
        self.task_env = runtime.task_env
        self.plant = runtime.plant
        self.network = network
        self.energy_management_policy = energy_management_policy
        self.charging = ChargingDynamics(charging_config)
        self.deterministic_energy_management = bool(deterministic_energy_management)
        self.action_space = runtime.action_space
        self.observation_space = self.task_env.observation_space
        self.certificate_provider: PersistentGoalCertificateProvider | None = None
        self.metrics = PersistentMetrics()
        self.energy_management_transitions: list[EnergyManagementTransition] = []
        self.last_energy_management_transition: EnergyManagementTransition | None = None
        self.last_requested_decision: EnergyDecision | None = None
        self.last_executed_decision: EnergyDecision | None = None
        self.last_override_reason: str | None = None
        self._decision_accumulator: _DecisionAccumulator | None = None
        self._charging_steps_since_decision = 0
        self._current_visit_steps = 0
        self._time_since_last_charge = 0
        self._last_departure_energy = 0.0
        self._active_task_start_step = 0

    @property
    def scheduler(self):
        """Deprecated alias for callers from the previous prototype."""
        return self.energy_management_policy

    @property
    def scheduler_transitions(self):
        """Deprecated alias for callers from the previous prototype."""
        return self.energy_management_transitions

    @property
    def manifest_hash(self) -> str:
        if self.certificate_provider is None:
            return "UNINITIALIZED"
        return self.certificate_provider.persistent_manifest.manifest_hash

    @property
    def configured_world_size(self) -> np.ndarray:
        return self.plant.config.world_size

    def _activate_current_edge(self) -> None:
        edge = self.task_env.manager.active_edge
        if edge is not None and self.certificate_provider is not None:
            self.certificate_provider.activate_edge(edge.edge_id)

    def _refresh_certificate_context(self) -> dict[str, Any]:
        self._activate_current_edge()
        context = self.runtime.preview_next_action_context()
        required = float(context.get("recovery_energy_required") or 0.0)
        reported_margin = context.get("energy_margin")
        margin = float(self.plant.state.energy - required if reported_margin is None else reported_margin)
        self.task_env.set_certificate_quantities(required, margin)
        return context

    def _departure_required(self) -> float:
        if self.certificate_provider is None:
            return float("inf")
        return self.certificate_provider.required_departure_energy(self.task_env.manager.current_task)

    def _departure_allowed(self) -> bool:
        return verify_departure_energy(
            self.plant.state.energy,
            self._departure_required(),
            self.charging.config.departure_energy_margin,
            self.certificate_provider is not None and self.certificate_provider.gate_pass,
        ).allowed

    def energy_management_observation(self) -> np.ndarray:
        task = self.task_env.manager.current_task
        state = self.plant.state
        station = self.plant.scenario.station_position
        goal = state.position if task is None else task.goal_position
        capacity = self.charging.config.battery_capacity
        required_departure = self._departure_required()
        estimated_goal_energy = 0.0
        estimated_goal_plus_return = 0.0
        if task is not None and self.certificate_provider is not None:
            try:
                from .persistent_task import GoalEdgeType

                if self.task_env.manager.current_node == self.network.charging_station:
                    estimated_goal_energy = self.certificate_provider.path_energy_upper(
                        self.network.charging_station,
                        task.goal_node,
                        GoalEdgeType.DEPARTURE_EDGE,
                    )
                else:
                    estimated_goal_energy = self.certificate_provider.path_energy_upper(
                        self.task_env.manager.current_node,
                        task.goal_node,
                        GoalEdgeType.TASK_EDGE,
                    )
                estimated_goal_plus_return = estimated_goal_energy + self.certificate_provider.path_energy_upper(
                    task.goal_node,
                    self.network.charging_station,
                    GoalEdgeType.RECOVERY_EDGE,
                )
            except ValueError:
                estimated_goal_energy = float("inf")
                estimated_goal_plus_return = float("inf")
        world_norm = float(np.linalg.norm(self.configured_world_size))
        values = np.asarray((
            state.energy / capacity,
            self.task_env.required_return_energy / capacity,
            self.task_env.energy_margin / capacity,
            np.linalg.norm(state.position - goal) / world_norm,
            np.linalg.norm(state.position - station) / world_norm,
            estimated_goal_energy / capacity if np.isfinite(estimated_goal_energy) else 2.0,
            estimated_goal_plus_return / capacity if np.isfinite(estimated_goal_plus_return) else 2.0,
            0.0 if task is None else task.reward / 10.0,
            self.task_env.manager.tasks_completed / 100.0,
            self._time_since_last_charge / max(1, self.plant.config.episode_limit),
            self._last_departure_energy / capacity,
            float(self.task_env.mode == PersistentMissionMode.CHARGING),
            required_departure / capacity if np.isfinite(required_departure) else 2.0,
        ), dtype=np.float32)
        return np.clip(values, -2.0, 2.0)

    def _energy_management_context(self) -> dict[str, Any]:
        return {
            "charging": self.task_env.mode == PersistentMissionMode.CHARGING,
            "departure_allowed": self._departure_allowed(),
            "energy_fraction": self.plant.state.energy / self.charging.config.battery_capacity,
            "energy_margin": self.task_env.energy_margin,
            "required_departure_energy": self._departure_required(),
            "scenario_id": self.plant.scenario.name,
            "manifest_hash": self.manifest_hash,
        }

    def _close_decision(
        self,
        *,
        terminated: bool,
        forced_override: bool = False,
        override_reason: str | None = None,
    ) -> None:
        accumulator = self._decision_accumulator
        if accumulator is None or accumulator.duration_steps <= 0:
            self._decision_accumulator = None
            return
        transition = EnergyManagementTransition(
            accumulator.observation,
            accumulator.requested,
            accumulator.executed,
            accumulator.cumulative_reward,
            accumulator.duration_steps,
            self.energy_management_observation(),
            terminated,
            forced_override,
            override_reason,
            self.plant.scenario.name,
            self.manifest_hash,
        )
        self.energy_management_transitions.append(transition)
        self.last_energy_management_transition = transition
        observe = getattr(self.energy_management_policy, "observe", None)
        if callable(observe):
            observe(transition)
        self._decision_accumulator = None

    def _start_decision(self, requested: EnergyDecision, executed: EnergyDecision) -> None:
        self.last_requested_decision = requested
        self.last_executed_decision = executed
        self._decision_accumulator = _DecisionAccumulator(
            self.energy_management_observation().copy(),
            int(requested),
            int(executed),
        )

    def _make_energy_decision(self) -> None:
        requested = EnergyDecision(self.energy_management_policy.select_action(
            self.energy_management_observation(),
            self._energy_management_context(),
            deterministic=self.deterministic_energy_management,
        ))
        executed = requested
        override_reason = None
        if self.task_env.mode == PersistentMissionMode.CHARGING:
            if requested == EnergyDecision.SERVE_OR_LEAVE:
                if self._departure_allowed():
                    self.task_env.leave_station()
                    self.task_env.manager.accept_service_decision()
                    self._activate_current_edge()
                    self._last_departure_energy = self.plant.state.energy
                    self._time_since_last_charge = 0
                    self.metrics.charge_durations.append(self._current_visit_steps)
                    self.metrics.energy_on_departure.append(float(self.plant.state.energy))
                    self._current_visit_steps = 0
                else:
                    executed = EnergyDecision.CHARGE_OR_STAY
                    override_reason = "INSUFFICIENT_DEPARTURE_ENERGY"
                    self.metrics.departure_rejection_count += 1
                    self.metrics.energy_management_override_count += 1
        elif requested == EnergyDecision.CHARGE_OR_STAY:
            self.task_env.request_return(forced=False)
            self.metrics.voluntary_return_count += 1
            self.metrics.energy_on_return_request.append(float(self.plant.state.energy))
        else:
            self.task_env.manager.accept_service_decision()
        self.last_override_reason = override_reason
        self._start_decision(requested, executed)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.runtime.mission_provider = None
        _, reset_info = self.runtime.reset(seed=seed, options=options)
        self.certificate_provider = PersistentGoalCertificateProvider(
            self.runtime,
            self.network,
            self.charging.config.battery_capacity,
        )
        self.runtime.mission_provider = self.certificate_provider
        self.certificate_provider.reset()
        replay = getattr(self.energy_management_policy, "replay", None)
        if replay is not None and hasattr(replay, "scenario_manifests"):
            replay.scenario_manifests[self.plant.scenario.name] = self.manifest_hash
        self.metrics = PersistentMetrics()
        self.energy_management_transitions.clear()
        self.last_energy_management_transition = None
        self._decision_accumulator = None
        self._charging_steps_since_decision = 0
        self._current_visit_steps = 0
        self._time_since_last_charge = 0
        self._last_departure_energy = self.plant.state.energy
        self._active_task_start_step = 0
        context = self._refresh_certificate_context()
        self._make_energy_decision()
        observation = self.task_env.build_observation(self.runtime._map_encoding(), self.runtime._corridor_encoding())
        return observation, reset_info | {
            "persistent_certificate_gate": "PASS" if self.certificate_provider.gate_pass else "blocked-by-persistent-certificate",
            "persistent_manifest_hash": self.manifest_hash,
            "action_context": context,
            "requested_mode": self.last_requested_decision.name,
            "executed_mode": self.last_executed_decision.name,
            "override_reason": self.last_override_reason,
        }

    def _accumulate(self, reward: float) -> None:
        if self._decision_accumulator is not None:
            self._decision_accumulator.cumulative_reward += float(reward)
            self._decision_accumulator.duration_steps += 1

    def _charging_step(self):
        result = self.charging.step(self.plant, self.manifest_hash)
        self._charging_steps_since_decision += 1
        self._current_visit_steps += 1
        self.metrics.charging_steps += 1
        self.metrics.total_steps += 1
        self.metrics.energy_charged += result.charged_energy
        reward = -self.task_env.reward_config.charging_dwell_cost
        self._accumulate(reward)
        if self._charging_steps_since_decision >= self.charging.config.checkpoint_steps:
            self._close_decision(terminated=result.truncated)
            self._charging_steps_since_decision = 0
            if not result.truncated:
                self._make_energy_decision()
        observation = self.task_env.build_observation(self.runtime._map_encoding(), self.runtime._corridor_encoding())
        return observation, reward, False, result.truncated, {
            "telemetry": result.telemetry,
            "persistent_mode": self.task_env.mode.name,
            "command_source": "charger_hold",
            "charged_energy": result.charged_energy,
            "requested_mode": None if self.last_requested_decision is None else self.last_requested_decision.name,
            "executed_mode": None if self.last_executed_decision is None else self.last_executed_decision.name,
            "override_reason": self.last_override_reason,
            "persistent_metrics": self.metric_snapshot(),
            "persistent_manifest_hash": self.manifest_hash,
        }

    def step(self, actor_output: np.ndarray):
        context = self._refresh_certificate_context()
        if (
            self.task_env.mode == PersistentMissionMode.TASK
            and self.task_env.energy_margin <= self.charging.config.forced_return_margin
        ):
            if self._decision_accumulator is not None:
                self._decision_accumulator.cumulative_reward -= self.task_env.reward_config.forced_return_interruption_cost
                self._decision_accumulator.executed = int(EnergyDecision.CHARGE_OR_STAY)
            self.task_env.request_return(forced=True)
            self._close_decision(
                terminated=False,
                forced_override=True,
                override_reason="ENERGY_MARGIN_FORCED_RETURN",
            )
            self.metrics.forced_return_count += 1
            self.metrics.energy_management_override_count += 1
            self.metrics.energy_on_return_request.append(float(self.plant.state.energy))
            self.last_executed_decision = EnergyDecision.CHARGE_OR_STAY
            self.last_override_reason = "ENERGY_MARGIN_FORCED_RETURN"
        if self.task_env.mode == PersistentMissionMode.CHARGING:
            return self._charging_step()

        returning_before = self.task_env.mode in {
            PersistentMissionMode.VOLUNTARY_RETURN,
            PersistentMissionMode.FORCED_RETURN,
        }
        if returning_before:
            observation, reward, terminated, truncated, info = self.runtime.step_recovery(
                "PERSISTENT_FORCED_RETURN"
                if self.task_env.mode == PersistentMissionMode.FORCED_RETURN
                else "PERSISTENT_VOLUNTARY_RETURN"
            )
            self.metrics.returning_steps += 1
        else:
            observation, reward, terminated, truncated, info = self.runtime.step(actor_output)
            self.metrics.serving_steps += 1
        telemetry = info["telemetry"]
        self.metrics.total_steps += 1
        self.metrics.energy_consumed += telemetry.energy_cost
        self._time_since_last_charge += 1
        self._accumulate(reward)
        if info.get("collision") or telemetry.collision:
            self.metrics.collision_count += 1
        if info.get("failure_reason") == "energy_depleted":
            self.metrics.energy_depletion_count += 1
        if info.get("accepted") and not info.get("action_context", {}).get("certificate_valid", False):
            self.metrics.uncertified_publication_count += 1
        if info.get("fallback_reason") == "RECOVERY_CERTIFICATE_INVALID":
            self.metrics.invalid_kappa_fallback_count += 1

        runtime_forced_return = bool(
            not returning_before
            and self.task_env.mode == PersistentMissionMode.FORCED_RETURN
            and not terminated
            and not truncated
        )
        if runtime_forced_return:
            if self._decision_accumulator is not None:
                self._decision_accumulator.cumulative_reward -= self.task_env.reward_config.forced_return_interruption_cost
                self._decision_accumulator.executed = int(EnergyDecision.CHARGE_OR_STAY)
            reason = info.get("fallback_reason") or "CERTIFICATE_RUNTIME_FORCED_RETURN"
            self._close_decision(terminated=False, forced_override=True, override_reason=reason)
            self.metrics.forced_return_count += 1
            self.metrics.energy_management_override_count += 1
            self.metrics.energy_on_return_request.append(float(self.plant.state.energy))
            self.last_executed_decision = EnergyDecision.CHARGE_OR_STAY
            self.last_override_reason = reason

        if self.task_env.mode == PersistentMissionMode.CHARGING and returning_before:
            self.metrics.charging_visits += 1
            self.metrics.energy_on_station_arrival.append(float(self.plant.state.energy))
            self._charging_steps_since_decision = 0
            self._close_decision(terminated=False)
            self._make_energy_decision()
        elif self.task_env.manager.decision_required and not terminated and not truncated:
            self._close_decision(terminated=False)
            self._make_energy_decision()
        if terminated or truncated:
            self._close_decision(terminated=True)

        manager = self.task_env.manager
        if info.get("task_completed_now"):
            self.metrics.task_completion_steps.append(self.metrics.total_steps - self._active_task_start_step)
            self._active_task_start_step = self.metrics.total_steps
        self.metrics.tasks_completed = manager.tasks_completed
        self.metrics.task_interruption_count = manager.task_interruption_count
        return observation, reward, terminated, truncated, info | {
            "requested_mode": None if self.last_requested_decision is None else self.last_requested_decision.name,
            "executed_mode": None if self.last_executed_decision is None else self.last_executed_decision.name,
            "override_reason": self.last_override_reason,
            "energy_margin": self.task_env.energy_margin,
            "required_return_energy": self.task_env.required_return_energy,
            "departure_required_energy": self._departure_required(),
            "persistent_certificate_gate": "PASS"
            if self.certificate_provider and self.certificate_provider.gate_pass
            else "blocked-by-persistent-certificate",
            "persistent_manifest_hash": self.manifest_hash,
            "persistent_metrics": asdict(self.metrics),
            "action_context": info.get("action_context", context),
            "persistent_metric_summary": self.metric_snapshot(),
        }

    def metric_snapshot(self) -> dict[str, Any]:
        result = asdict(self.metrics)
        total = max(1, self.metrics.total_steps)
        returns = self.metrics.voluntary_return_count + self.metrics.forced_return_count
        result.update({
            "tasks_per_1000_steps": 1000.0 * self.metrics.tasks_completed / total,
            "mean_task_completion_steps": float(np.mean(self.metrics.task_completion_steps))
            if self.metrics.task_completion_steps else 0.0,
            "mean_charge_duration": float(np.mean(self.metrics.charge_durations))
            if self.metrics.charge_durations else 0.0,
            "forced_return_rate": self.metrics.forced_return_count / max(1, returns),
            "energy_management_override_rate": self.metrics.energy_management_override_count / total,
            "serving_fraction": self.metrics.serving_steps / total,
            "returning_fraction": self.metrics.returning_steps / total,
            "charging_fraction": self.metrics.charging_steps / total,
        })
        return result


class PersistentRuntimeWrapper(gym.Env[np.ndarray, np.ndarray]):
    """Main one-policy persistent runtime with certified κ backup only."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        runtime: CertifiedRuntimeWrapper,
        network: CertifiedGoalNetwork,
        charging_config: ChargingConfig | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(runtime.task_env, PersistentGoalWrapper):
            raise TypeError("persistent runtime requires PersistentGoalWrapper")
        if runtime.config.terminate_on_terminal:
            raise ValueError("persistent plant must set terminate_on_terminal=False")
        if runtime.generator_center_mode not in {"zero", "safety_neutral"}:
            raise ValueError("main persistent method requires a safety-neutral Generator center")
        self.runtime = runtime
        self.task_env = runtime.task_env
        self.plant = runtime.plant
        self.network = network
        self.charging = ChargingDynamics(charging_config)
        self.action_space = runtime.action_space
        self.observation_space = self.task_env.observation_space
        self.certificate_provider: PersistentGoalCertificateProvider | None = None
        self.metrics = PersistentMetrics()
        self.policy_authority_certificate = None
        self._last_authority_decision: PersistentAuthorityDecision | None = None
        self._active_task_start_step = 0
        self._time_since_last_charge = 0
        self._station_approach_active = False

    @property
    def manifest_hash(self) -> str:
        if self.certificate_provider is None:
            return "UNINITIALIZED"
        return self.certificate_provider.persistent_manifest.manifest_hash

    @property
    def trainable_policy_count(self) -> int:
        return 1

    def _activate_current_edge(self) -> None:
        edge = self.task_env.manager.active_edge
        if edge is not None and self.certificate_provider is not None:
            self.certificate_provider.activate_edge(edge.edge_id)

    def _refresh_context(self) -> dict[str, Any]:
        self._activate_current_edge()
        charging_state = self.task_env.mode == PersistentMissionMode.CHARGING_RL
        departure = self._departure_gate() if charging_state else DepartureGateResult(True, 0.0, None)
        if self.certificate_provider is not None:
            self.certificate_provider.configure_charging_support(charging_state and not departure.allowed)
        context = self.runtime.preview_next_action_context()
        required = float(context.get("recovery_energy_required") or float("inf"))
        margin = float(context.get("energy_margin") if context.get("energy_margin") is not None else float("-inf"))
        self.task_env.set_certificate_quantities(required, margin)
        self.task_env.set_time_since_last_charge(self._time_since_last_charge)
        self.metrics.minimum_energy_margin = min(self.metrics.minimum_energy_margin, margin)
        if self.certificate_provider is not None and self.certificate_provider.last_context is not None:
            task = self.task_env.manager.current_task
            goal = self.plant.state.position if task is None else task.goal_position
            verifier = self.certificate_provider.recoverability_verifiers[
                self.certificate_provider.active_edge_id
            ]
            self.policy_authority_certificate = verifier.policy_authority(
                self.runtime._certificate_state(),
                self.certificate_provider.last_context,
                goal,
                self.plant.scenario.station_position,
            )
            station_hold_valid = bool(verifier.certified_station_hold(self.runtime._certificate_state())) if charging_state and not departure.allowed else False
        else:
            self.policy_authority_certificate = None
            station_hold_valid = False
        policy_authority_pass = bool(
            self.policy_authority_certificate is not None
            and (
                self.policy_authority_certificate.passed
                or (
                    charging_state
                    and not departure.allowed
                    and self.policy_authority_certificate.neutral_center
                    and self.policy_authority_certificate.full_rank
                    and self.policy_authority_certificate.nondegenerate
                    and self.policy_authority_certificate.complete_set_recoverable
                )
            )
        )
        authority_input = PersistentAuthorityInput(
            persistent_mode=self.task_env.mode.name,
            energy_margin=margin,
            backup_switch_margin=float(self.charging.config.forced_return_margin),
            persistent_certificate_valid=bool(self.certificate_provider is not None and self.certificate_provider.gate_pass),
            certificate_valid=bool(context.get("certificate_valid", False)),
            kappa_valid=bool(context.get("certificate_valid", False) and context.get("kappa") is not None),
            generator_available=bool(context.get("generator_available", False)),
            recoverable_set_member=context.get("recoverable_set_member") is True,
            recoverability_action_verified=context.get("recoverability_action_verified") is True,
            policy_authority_pass=policy_authority_pass,
            charging_state=charging_state,
            departure_allowed=bool(departure.allowed),
            charging_support_verified=context.get("charging_support_verified") is True,
            station_hold_valid=station_hold_valid,
        )
        self._last_authority_decision = PersistentExecutionAuthority.evaluate(authority_input)
        return context | {
            "persistent_mode": self.task_env.mode.name,
            "persistent_certificate_valid": authority_input.persistent_certificate_valid,
            "policy_authority_pass": authority_input.policy_authority_pass,
            "departure_allowed": authority_input.departure_allowed,
            "departure_reason": departure.reason,
            "station_hold_valid": station_hold_valid,
            "execution_authority": self._last_authority_decision.authority.value,
            "execution_authority_reason": self._last_authority_decision.reason,
            "generator_executable": self._last_authority_decision.generator_executable,
            "backup_required": self._last_authority_decision.kappa_required,
            "backup_switch_margin": authority_input.backup_switch_margin,
            "charging_restriction": self._last_authority_decision.charging_restriction,
            "station_hold_required": self._last_authority_decision.station_hold_required,
        }

    def _departure_required(self) -> float:
        if self.certificate_provider is None:
            return float("inf")
        return self.certificate_provider.required_departure_energy(self.task_env.manager.current_task)

    def _departure_gate(self) -> DepartureGateResult:
        return verify_departure_energy(
            self.plant.state.energy,
            self._departure_required(),
            self.charging.config.departure_energy_margin,
            self.certificate_provider is not None and self.certificate_provider.gate_pass,
        )

    @staticmethod
    def _candidate_from_context(actor_u: np.ndarray, context: dict[str, Any]) -> np.ndarray | None:
        selected = np.asarray(actor_u, dtype=np.float64)
        if selected.shape != (3,) or not np.all(np.isfinite(selected)):
            return None
        if not context.get("generator_available") or context.get("c") is None or context.get("G") is None:
            return None
        return np.asarray(context["c"], dtype=np.float64) + np.asarray(context["G"], dtype=np.float64) @ np.tanh(selected)

    def _backup_reason(self, context: dict[str, Any]) -> str | None:
        if not context.get("certificate_valid", False) and context.get("failure_reason"):
            return str(context["failure_reason"])
        if self._last_authority_decision is None:
            self._last_authority_decision = PersistentExecutionAuthority.evaluate(PersistentAuthorityInput(
                persistent_mode=self.task_env.mode.name,
                energy_margin=float(self.task_env.energy_margin),
                backup_switch_margin=float(self.charging.config.forced_return_margin),
                persistent_certificate_valid=bool(self.certificate_provider is not None and self.certificate_provider.gate_pass),
                certificate_valid=bool(context.get("certificate_valid", False)),
                kappa_valid=bool(context.get("certificate_valid", False)),
                generator_available=bool(context.get("generator_available", False)),
                recoverable_set_member=context.get("recoverable_set_member") is True,
                recoverability_action_verified=context.get("recoverability_action_verified") is True,
                policy_authority_pass=bool(self.policy_authority_certificate is not None and self.policy_authority_certificate.passed),
                charging_state=False,
                departure_allowed=True,
                charging_support_verified=False,
                station_hold_valid=False,
            ))
        if self._last_authority_decision.authority in {ExecutionAuthority.KAPPA_BACKUP, ExecutionAuthority.FAIL_CLOSED}:
            if self._last_authority_decision.reason == "RECOVERY_CERTIFICATE_INVALID" and context.get("failure_reason"):
                return str(context["failure_reason"])
            return self._last_authority_decision.reason
        return None

    def _begin_backup(self, reason: str) -> None:
        if self.task_env.mode != PersistentMissionMode.BACKUP_RECOVERY:
            self.metrics.backup_recovery_count += 1
            self.metrics.forced_return_count += 1
            self.metrics.energy_margin_at_backup.append(float(self.task_env.energy_margin))
        self.task_env.begin_backup_recovery(reason)

    def _station_hold_step(self, reason: str):
        if self.certificate_provider is None:
            raise RuntimeError("persistent certificate provider is unavailable")
        state = self.runtime._certificate_state()
        if not self.certificate_provider.certified_station_hold(state):
            observation = self.task_env.build_observation(self.runtime._map_encoding(), self.runtime._corridor_encoding())
            self.task_env.mode = PersistentMissionMode.FAILURE
            self.task_env.phase = self.task_env.mode
            return observation, 0.0, True, False, {
                "failure_reason": "CERTIFIED_CHARGER_HOLD_UNAVAILABLE",
                "departure_rejected": True,
                "departure_rejection_reason": reason,
                "command_source": "none",
            }
        result = self.charging.step(self.plant, self.manifest_hash)
        self.task_env.episode_step += 1
        self.metrics.total_steps += 1
        self.metrics.charging_steps += 1
        self.metrics.energy_charged += result.charged_energy
        self.metrics.departure_rejection_count += 1
        self._time_since_last_charge = 0
        self.task_env.set_time_since_last_charge(0)
        reward = -self.task_env.reward_config.charging_dwell_cost
        observation = self.task_env.build_observation(self.runtime._map_encoding(), self.runtime._corridor_encoding())
        return observation, reward, False, result.truncated, {
            "telemetry": result.telemetry,
            "accepted": False,
            "fallback_reason": reason,
            "backup_triggered": False,
            "backup_reason": None,
            "critic_action": np.zeros(3, dtype=np.float64),
            "command_source": "charger_hold",
            "execution_authority": ExecutionAuthority.CHARGER_CONSTRAINED.value,
            "execution_authority_reason": reason,
            "departure_attempt": True,
            "departure_rejected": True,
            "departure_rejection_reason": reason,
            "persistent_mode": self.task_env.mode.name,
            "persistent_manifest_hash": self.manifest_hash,
            "persistent_metrics": self.metric_snapshot(),
        }

    def _apply_charging(self, info: dict[str, Any], reward: float):
        telemetry = info["telemetry"]
        result = self.charging.apply_during_motion_cycle(self.plant, telemetry)
        if result.charged_energy <= 0.0:
            return info, reward
        self.metrics.energy_charged += result.charged_energy
        self.metrics.charging_steps += 1
        self._time_since_last_charge = 0
        self.task_env.set_time_since_last_charge(0)
        return info | {"telemetry": result.telemetry, "charged_energy": result.charged_energy}, reward - self.task_env.reward_config.charging_dwell_cost

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.runtime.mission_provider = None
        _, reset_info = self.runtime.reset(seed=seed, options=options)
        self.certificate_provider = PersistentGoalCertificateProvider(
            self.runtime,
            self.network,
            self.charging.config.battery_capacity,
        )
        self.runtime.mission_provider = self.certificate_provider
        self.certificate_provider.reset()
        self.metrics = PersistentMetrics()
        self._active_task_start_step = 0
        self._time_since_last_charge = 0
        self._station_approach_active = False
        self._last_authority_decision = None
        context = self._refresh_context()
        observation = self.task_env.build_observation(self.runtime._map_encoding(), self.runtime._corridor_encoding())
        return observation, reset_info | {
            "persistent_certificate_gate": "PASS" if self.certificate_provider.gate_pass else "blocked-by-persistent-certificate",
            "policy_authority_gate": "PASS" if self.policy_authority_certificate.passed else "FAIL",
            "persistent_manifest_hash": self.manifest_hash,
            "action_context": context,
            "trainable_policy_count": self.trainable_policy_count,
        }

    def step(self, actor_output: np.ndarray):
        actor_u = np.asarray(actor_output, dtype=np.float64)
        if actor_u.shape != (3,):
            raise ValueError("persistent Generator policy output must have shape (3,)")
        mode_before = self.task_env.mode
        context = self._refresh_context()
        decision = self._last_authority_decision
        if decision is None:
            raise RuntimeError("persistent execution authority was not evaluated")
        backup_reason = self._backup_reason(context)
        if decision.authority in {ExecutionAuthority.KAPPA_BACKUP, ExecutionAuthority.FAIL_CLOSED}:
            backup_reason = decision.reason
            self._begin_backup(backup_reason)
            observation, reward, terminated, truncated, info = self.runtime.step_recovery(backup_reason)
        elif decision.authority == ExecutionAuthority.CHARGER_CONSTRAINED and decision.station_hold_required:
            return self._station_hold_step(decision.reason)
        else:
            candidate = self._candidate_from_context(actor_u, context)
            if candidate is None:
                self._begin_backup("ACTOR_OR_GENERATOR_INVALID")
                observation, reward, terminated, truncated, info = self.runtime.step_recovery("ACTOR_OR_GENERATOR_INVALID")
                backup_reason = "ACTOR_OR_GENERATOR_INVALID"
            else:
                observation, reward, terminated, truncated, info = self.runtime.step(actor_u)

        telemetry = info["telemetry"]
        actual_authority = decision.authority
        actual_authority_reason = decision.reason
        if not info.get("accepted", False):
            runtime_reason = info.get("fallback_reason")
            if runtime_reason is not None and backup_reason is None:
                backup_reason = str(runtime_reason)
                self._begin_backup(backup_reason)
            if backup_reason is not None:
                actual_authority = ExecutionAuthority.KAPPA_BACKUP
                actual_authority_reason = backup_reason
        self.metrics.total_steps += 1
        self.metrics.energy_consumed += telemetry.energy_cost
        self._time_since_last_charge += 1
        self.task_env.set_time_since_last_charge(self._time_since_last_charge)
        if info.get("accepted"):
            self.metrics.generator_accepted_steps += 1
        else:
            self.metrics.no_generator_steps += int(info.get("fallback_reason") == "NO_GENERATOR_SET")
        if telemetry.collision:
            self.metrics.collision_count += 1
        if info.get("failure_reason") == "energy_depleted":
            self.metrics.energy_depletion_count += 1
        if info.get("accepted") and not info.get("action_context", {}).get("recoverability_action_verified", False):
            self.metrics.uncertified_publication_count += 1
        if info.get("fallback_reason") == "RECOVERY_CERTIFICATE_INVALID":
            self.metrics.invalid_kappa_fallback_count += 1

        terminal_now = self.plant.terminal.is_charge_admissible(self.plant.state)
        was_charging = mode_before == PersistentMissionMode.CHARGING_RL
        left_station = False
        if self.task_env.mode == PersistentMissionMode.TASK_RL and terminal_now and backup_reason is None:
            self.task_env.enter_charging(voluntary=True)
            self.metrics.voluntary_station_arrivals += 1
            self.metrics.voluntary_return_count += 1
            self.metrics.charging_visits += 1
            self.metrics.energy_margin_at_station_approach.append(float(self.task_env.energy_margin))
            self.metrics.energy_on_station_arrival.append(float(self.plant.state.energy))
            self._station_approach_active = False
        elif mode_before == PersistentMissionMode.BACKUP_RECOVERY and self.task_env.mode == PersistentMissionMode.CHARGING_RL and terminal_now:
            self.metrics.charging_visits += 1
            self.metrics.energy_on_station_arrival.append(float(self.plant.state.energy))
        elif self.task_env.mode == PersistentMissionMode.CHARGING_RL and was_charging and not terminal_now and info.get("accepted"):
            self.task_env.leave_station()
            self.metrics.energy_on_departure.append(float(self.plant.state.energy))
            self._time_since_last_charge = 0
            left_station = True

        station_distance = float(np.linalg.norm(self.plant.state.position - self.plant.scenario.station_position))
        approach_radius = 3.0 * float(np.max(self.plant.scenario.terminal.position_high - self.plant.scenario.terminal.position_low))
        if (
            self.task_env.mode == PersistentMissionMode.TASK_RL
            and backup_reason is None
            and station_distance <= approach_radius
        ):
            self._station_approach_active = True
            self.task_env.voluntary_station_approach = True

        if self.task_env.mode == PersistentMissionMode.CHARGING_RL and terminal_now:
            info, reward = self._apply_charging(info, reward)
            observation = self.task_env.build_observation(self.runtime._map_encoding(), self.runtime._corridor_encoding())
        elif left_station:
            observation = self.task_env.build_observation(self.runtime._map_encoding(), self.runtime._corridor_encoding())
        if backup_reason is not None:
            reward -= self.task_env.reward_config.backup_intervention_cost
        if info.get("task_completed_now"):
            self.metrics.task_completion_steps.append(self.metrics.total_steps - self._active_task_start_step)
            self._active_task_start_step = self.metrics.total_steps
        self.metrics.tasks_completed = self.task_env.manager.tasks_completed
        self.metrics.task_interruption_count = self.task_env.manager.task_interruption_count
        return observation, reward, terminated, truncated, info | {
            "persistent_mode": self.task_env.mode.name,
            "backup_triggered": backup_reason is not None,
            "backup_reason": backup_reason,
            "voluntary_station_approach": self._station_approach_active,
            "voluntary_station_arrival": self.metrics.voluntary_station_arrivals > 0 and terminal_now and backup_reason is None,
            "charging": self.task_env.mode == PersistentMissionMode.CHARGING_RL,
            "departure_attempt": was_charging and not terminal_now,
            "departure_rejected": False,
            "departure_rejection_reason": None,
            "required_return_energy": self.task_env.required_return_energy,
            "energy_margin": self.task_env.energy_margin,
            "persistent_manifest_hash": self.manifest_hash,
            "execution_authority": actual_authority.value,
            "execution_authority_reason": actual_authority_reason,
            "generator_executable": decision.generator_executable and actual_authority in {
                ExecutionAuthority.RL_GENERATOR,
                ExecutionAuthority.CHARGER_CONSTRAINED,
            },
            "charging_restriction": decision.charging_restriction,
            "persistent_metrics": self.metric_snapshot(),
        }

    def metric_snapshot(self) -> dict[str, Any]:
        result = asdict(self.metrics)
        total = max(1, self.metrics.total_steps)
        finite_margin = self.metrics.minimum_energy_margin if np.isfinite(self.metrics.minimum_energy_margin) else None
        result.update({
            "tasks_per_1000_steps": 1000.0 * self.metrics.tasks_completed / total,
            "mean_goal_completion_time": float(np.mean(self.metrics.task_completion_steps)) if self.metrics.task_completion_steps else 0.0,
            "energy_per_task": self.metrics.energy_consumed / max(1, self.metrics.tasks_completed),
            "backup_rate": self.metrics.backup_recovery_count / total,
            "generator_acceptance_rate": self.metrics.generator_accepted_steps / total,
            "no_generator_rate": self.metrics.no_generator_steps / total,
            "charging_fraction": self.metrics.charging_steps / total,
            "minimum_energy_margin": finite_margin,
        })
        return result
