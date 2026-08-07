from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np

from cert_runtime.charging_scheduler import (
    ChargingScheduler,
    SchedulerBinaryDecision,
    SchedulerTransition,
)

from .charging import ChargingConfig, ChargingDynamics, verify_departure_energy
from .persistent_certificate import PersistentCertificateProvider
from .persistent_task import (
    CertifiedServiceNetwork,
    PersistentMissionMode,
    PersistentTaskWrapper,
)
from .runtime_wrapper import CertifiedRuntimeWrapper


@dataclass(slots=True)
class PersistentMetrics:
    tasks_completed: int = 0
    pickup_count: int = 0
    delivery_count: int = 0
    energy_consumed: float = 0.0
    energy_charged: float = 0.0
    charging_visits: int = 0
    total_charging_steps: int = 0
    voluntary_return_count: int = 0
    forced_return_count: int = 0
    scheduler_override_count: int = 0
    departure_rejection_count: int = 0
    task_pause_count: int = 0
    task_resume_count: int = 0
    serving_steps: int = 0
    returning_steps: int = 0
    collision_count: int = 0
    energy_depletion_count: int = 0
    uncertified_publication_count: int = 0
    invalid_kappa_fallback_count: int = 0
    total_steps: int = 0
    energy_on_charge_request: list[float] = field(default_factory=list)
    energy_on_station_arrival: list[float] = field(default_factory=list)
    energy_on_departure: list[float] = field(default_factory=list)
    charging_durations: list[int] = field(default_factory=list)
    task_completion_latencies: list[int] = field(default_factory=list)


@dataclass(slots=True)
class _DecisionAccumulator:
    observation: np.ndarray
    requested: int
    executed: int
    cumulative_reward: float = 0.0
    duration_steps: int = 0


class PersistentRuntimeWrapper(gym.Env[np.ndarray, np.ndarray]):
    """Event-driven charging scheduler around the unchanged certificate runtime.

    Continuous motion remains owned by ``CertifiedRuntimeWrapper``.  This layer
    selects service/charge modes, applies the hard energy and departure gates,
    and emits decision-to-decision SMDP records.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        runtime: CertifiedRuntimeWrapper,
        network: CertifiedServiceNetwork,
        scheduler: ChargingScheduler,
        charging_config: ChargingConfig | None = None,
        deterministic_scheduler: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(runtime.task_env, PersistentTaskWrapper):
            raise TypeError("persistent runtime requires PersistentTaskWrapper")
        if runtime.config.terminate_on_terminal:
            raise ValueError("persistent plant must set terminate_on_terminal=False")
        self.runtime = runtime
        self.task_env = runtime.task_env
        self.plant = runtime.plant
        self.network = network
        self.scheduler = scheduler
        self.charging = ChargingDynamics(charging_config)
        self.deterministic_scheduler = bool(deterministic_scheduler)
        self.action_space = runtime.action_space
        self.observation_space = self.task_env.observation_space
        self.certificate_provider: PersistentCertificateProvider | None = None
        self.metrics = PersistentMetrics()
        self.scheduler_transitions: list[SchedulerTransition] = []
        self.last_scheduler_transition: SchedulerTransition | None = None
        self.last_requested_decision: SchedulerBinaryDecision | None = None
        self.last_executed_decision: SchedulerBinaryDecision | None = None
        self.last_override_reason: str | None = None
        self._decision_accumulator: _DecisionAccumulator | None = None
        self._charging_steps_since_decision = 0
        self._charging_visit_steps: list[int] = []
        self._current_visit_steps = 0
        self._time_since_last_charge = 0
        self._last_charge_energy = 0.0
        self._active_task_start_step = 0

    @property
    def manifest_hash(self) -> str:
        if self.certificate_provider is None:
            return "UNINITIALIZED"
        return self.certificate_provider.persistent_manifest.manifest_hash

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
        return self.certificate_provider.required_departure_energy(
            self.task_env.manager.current_task,
            self.task_env.manager.paused_status,
        )

    def _departure_allowed(self) -> bool:
        return verify_departure_energy(
            self.plant.state.energy,
            self._departure_required(),
            self.charging.config.departure_energy_margin,
            self.certificate_provider is not None and self.certificate_provider.gate_pass,
        ).allowed

    def scheduler_observation(self) -> np.ndarray:
        task = self.task_env.manager.current_task
        state = self.plant.state
        station = self.plant.scenario.station_position
        pickup = state.position if task is None else task.pickup_position
        dropoff = state.position if task is None else task.dropoff_position
        required_departure = self._departure_required()
        estimated_task_energy = 0.0
        if task is not None and self.certificate_provider is not None:
            status = self.task_env.manager.paused_status or task.status
            if status.name == "TO_PICKUP":
                estimated_task_energy = (
                    self.certificate_provider.path_energy_upper(self.network.charging_station, task.pickup_node)
                    + self.certificate_provider.path_energy_upper(task.pickup_node, task.dropoff_node)
                )
            else:
                estimated_task_energy = self.certificate_provider.path_energy_upper(self.network.charging_station, task.dropoff_node)
        capacity = self.charging.config.battery_capacity
        values = np.asarray(
            (
                state.energy / capacity,
                self.task_env.required_return_energy / capacity,
                self.task_env.energy_margin / capacity,
                np.linalg.norm(state.position - station) / np.linalg.norm(self.configured_world_size),
                np.linalg.norm(state.position - pickup) / np.linalg.norm(self.configured_world_size),
                np.linalg.norm(pickup - dropoff) / np.linalg.norm(self.configured_world_size),
                0.0 if task is None else task.reward / 10.0,
                0.0 if task is None or task.deadline_optional is None else task.deadline_optional / 1000.0,
                estimated_task_energy / capacity,
                required_departure / capacity if np.isfinite(required_departure) else 2.0,
                self.task_env.manager.tasks_completed / 100.0,
                self._time_since_last_charge / max(1, self.plant.config.episode_limit),
                self._last_charge_energy / capacity,
                float(self.task_env.mode == PersistentMissionMode.CHARGING),
                state.energy / capacity,
            ),
            dtype=np.float32,
        )
        return np.clip(values, -2.0, 2.0)

    @property
    def configured_world_size(self) -> np.ndarray:
        return self.plant.config.world_size

    def _scheduler_context(self) -> dict[str, Any]:
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
        transition = SchedulerTransition(
            accumulator.observation,
            accumulator.requested,
            accumulator.executed,
            accumulator.cumulative_reward,
            accumulator.duration_steps,
            self.scheduler_observation(),
            terminated,
            forced_override,
            override_reason,
            self.plant.scenario.name,
            self.manifest_hash,
        )
        self.scheduler_transitions.append(transition)
        self.last_scheduler_transition = transition
        observe = getattr(self.scheduler, "observe", None)
        if callable(observe):
            observe(transition)
        self._decision_accumulator = None

    def _start_decision(self, requested: SchedulerBinaryDecision, executed: SchedulerBinaryDecision) -> None:
        self.last_requested_decision = requested
        self.last_executed_decision = executed
        self._decision_accumulator = _DecisionAccumulator(
            self.scheduler_observation().copy(),
            int(requested),
            int(executed),
        )

    def _make_scheduler_decision(self) -> None:
        observation = self.scheduler_observation()
        requested = SchedulerBinaryDecision(
            self.scheduler.select_action(
                observation,
                self._scheduler_context(),
                deterministic=self.deterministic_scheduler,
            )
        )
        executed = requested
        override_reason = None
        if self.task_env.mode == PersistentMissionMode.CHARGING:
            if requested == SchedulerBinaryDecision.SERVE_OR_LEAVE:
                if self._departure_allowed():
                    self.task_env.leave_station()
                    self.task_env.manager.accept_service_decision()
                    self._activate_current_edge()
                    self._last_charge_energy = self.plant.state.energy
                    self._time_since_last_charge = 0
                    self._charging_visit_steps.append(self._current_visit_steps)
                    self.metrics.charging_durations.append(self._current_visit_steps)
                    self.metrics.energy_on_departure.append(float(self.plant.state.energy))
                    self._current_visit_steps = 0
                else:
                    executed = SchedulerBinaryDecision.CHARGE_OR_STAY
                    override_reason = "INSUFFICIENT_DEPARTURE_ENERGY"
                    self.metrics.departure_rejection_count += 1
                    self.metrics.scheduler_override_count += 1
        else:
            if requested == SchedulerBinaryDecision.CHARGE_OR_STAY:
                self.task_env.request_return(forced=False)
                self.metrics.voluntary_return_count += 1
                self.metrics.energy_on_charge_request.append(float(self.plant.state.energy))
            else:
                self.task_env.manager.accept_service_decision()
        self.last_requested_decision = requested
        self.last_executed_decision = executed
        self.last_override_reason = override_reason
        self._start_decision(requested, executed)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.runtime.mission_provider = None
        _, reset_info = self.runtime.reset(seed=seed, options=options)
        self.certificate_provider = PersistentCertificateProvider(
            self.runtime,
            self.network,
            self.charging.config.battery_capacity,
        )
        self.runtime.mission_provider = self.certificate_provider
        self.certificate_provider.reset()
        replay = getattr(self.scheduler, "replay", None)
        if replay is not None and hasattr(replay, "scenario_manifests"):
            replay.scenario_manifests[self.plant.scenario.name] = self.manifest_hash
        self.metrics = PersistentMetrics()
        self.scheduler_transitions.clear()
        self.last_scheduler_transition = None
        self._decision_accumulator = None
        self._charging_steps_since_decision = 0
        self._charging_visit_steps.clear()
        self._current_visit_steps = 0
        self._time_since_last_charge = 0
        self._last_charge_energy = self.plant.state.energy
        self._active_task_start_step = 0
        context = self._refresh_certificate_context()
        self._make_scheduler_decision()
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
        epoch = self.manifest_hash
        result = self.charging.step(self.plant, epoch)
        self._charging_steps_since_decision += 1
        self._current_visit_steps += 1
        self.metrics.total_charging_steps += 1
        self.metrics.total_steps += 1
        self.metrics.energy_charged += result.charged_energy
        reward = -self.task_env.reward_config.charging_dwell_cost
        self._accumulate(reward)
        if self._charging_steps_since_decision >= self.charging.config.checkpoint_steps:
            self._close_decision(terminated=result.truncated)
            self._charging_steps_since_decision = 0
            if not result.truncated:
                self._make_scheduler_decision()
        observation = self.task_env.build_observation(self.runtime._map_encoding(), self.runtime._corridor_encoding())
        info = {
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
        return observation, reward, False, result.truncated, info

    def step(self, actor_output: np.ndarray):
        context = self._refresh_certificate_context()
        if (
            self.task_env.mode in {PersistentMissionMode.TO_PICKUP, PersistentMissionMode.TO_DROPOFF}
            and self.task_env.energy_margin <= self.charging.config.forced_return_margin
        ):
            if self._decision_accumulator is not None:
                self._decision_accumulator.cumulative_reward -= self.task_env.reward_config.forced_return_interruption_cost
                self._decision_accumulator.executed = int(SchedulerBinaryDecision.CHARGE_OR_STAY)
            self.task_env.request_return(forced=True)
            self._close_decision(
                terminated=False,
                forced_override=True,
                override_reason="ENERGY_MARGIN_FORCED_RETURN",
            )
            self.metrics.forced_return_count += 1
            self.metrics.scheduler_override_count += 1
            self.metrics.energy_on_charge_request.append(float(self.plant.state.energy))
            self.last_executed_decision = SchedulerBinaryDecision.CHARGE_OR_STAY
            self.last_override_reason = "ENERGY_MARGIN_FORCED_RETURN"
        if self.task_env.mode == PersistentMissionMode.CHARGING:
            return self._charging_step()
        returning = self.task_env.mode in {PersistentMissionMode.VOLUNTARY_RETURN, PersistentMissionMode.FORCED_RETURN}
        if returning:
            observation, reward, terminated, truncated, info = self.runtime.step_recovery(
                "PERSISTENT_FORCED_RETURN" if self.task_env.mode == PersistentMissionMode.FORCED_RETURN else "PERSISTENT_VOLUNTARY_RETURN"
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
            not returning
            and self.task_env.mode == PersistentMissionMode.FORCED_RETURN
            and not terminated
            and not truncated
        )
        if runtime_forced_return:
            if self._decision_accumulator is not None:
                self._decision_accumulator.cumulative_reward -= self.task_env.reward_config.forced_return_interruption_cost
                self._decision_accumulator.executed = int(SchedulerBinaryDecision.CHARGE_OR_STAY)
            reason = info.get("fallback_reason") or "CERTIFICATE_RUNTIME_FORCED_RETURN"
            self._close_decision(terminated=False, forced_override=True, override_reason=reason)
            self.metrics.forced_return_count += 1
            self.metrics.scheduler_override_count += 1
            self.metrics.energy_on_charge_request.append(float(self.plant.state.energy))
            self.last_executed_decision = SchedulerBinaryDecision.CHARGE_OR_STAY
            self.last_override_reason = reason
        if self.task_env.mode == PersistentMissionMode.CHARGING and returning:
            self.metrics.charging_visits += 1
            self.metrics.energy_on_station_arrival.append(float(self.plant.state.energy))
            self._charging_steps_since_decision = 0
            self._close_decision(terminated=False)
            self._make_scheduler_decision()
        elif self.task_env.manager.decision_required and not terminated and not truncated:
            self._close_decision(terminated=False)
            self._make_scheduler_decision()
        if terminated or truncated:
            self._close_decision(terminated=True)
        manager = self.task_env.manager
        if info.get("delivery_now"):
            self.metrics.task_completion_latencies.append(self.metrics.total_steps - self._active_task_start_step)
            self._active_task_start_step = self.metrics.total_steps
        self.metrics.tasks_completed = manager.tasks_completed
        self.metrics.pickup_count = manager.pickup_count
        self.metrics.delivery_count = manager.delivery_count
        self.metrics.task_pause_count = manager.task_pause_count
        self.metrics.task_resume_count = manager.task_resume_count
        return observation, reward, terminated, truncated, info | {
            "requested_mode": None if self.last_requested_decision is None else self.last_requested_decision.name,
            "executed_mode": None if self.last_executed_decision is None else self.last_executed_decision.name,
            "override_reason": self.last_override_reason,
            "energy_margin": self.task_env.energy_margin,
            "required_return_energy": self.task_env.required_return_energy,
            "departure_required_energy": self._departure_required(),
            "persistent_certificate_gate": "PASS" if self.certificate_provider and self.certificate_provider.gate_pass else "blocked-by-persistent-certificate",
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
            "mean_charge_duration": float(np.mean(self.metrics.charging_durations)) if self.metrics.charging_durations else 0.0,
            "forced_return_rate": self.metrics.forced_return_count / max(1, returns),
            "scheduler_override_rate": self.metrics.scheduler_override_count / total,
            "fraction_time_serving": self.metrics.serving_steps / total,
            "fraction_time_returning": self.metrics.returning_steps / total,
            "fraction_time_charging": self.metrics.total_charging_steps / total,
            "mean_task_completion_latency": float(np.mean(self.metrics.task_completion_latencies)) if self.metrics.task_completion_latencies else 0.0,
        })
        return result
