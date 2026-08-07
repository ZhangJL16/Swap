# Persistent Task Assignment and Autonomous Charging

## Status and scope

This module is an implementation scaffold with deterministic unit coverage. The persistent
certificate gate, environment acceptance, scheduler baselines, scheduler training, and evaluation
scripts have **not** been run in this development round. All charging and physical-bound values are
synthetic. No real-flight, physical-calibration, HIL, or WCET claim follows from this code.

The pre-existing single-mission environments and mission gates remain unchanged. Persistent
fixtures use separate files: `persistent_open`, `persistent_obstacle`, and
`persistent_energy_tight`.

## Architecture

`PersistentTaskManager` samples only declared directed edges of a finite
`CertifiedServiceNetwork`. A task contains an ID, pickup, dropoff, reward, optional deadline, and
one of `PENDING`, `TO_PICKUP`, `CARRYING`, `COMPLETED`, or `PAUSED_FOR_CHARGE`. Delivery assigns the
next task and does not terminate the episode. A charge interruption preserves the task and resumes
the same pickup/dropoff obligation.

`PersistentTaskWrapper` owns task state and reward. `CertifiedRuntimeWrapper` remains the sole
continuous-action certificate path: it stages independently certified kappa first, verifies the
Generator set, and passes only `a_exec` to the unchanged plant. `PersistentRuntimeWrapper` composes
the event-level scheduler, hard energy override, departure gate, certified return calls, and
charging dynamics around those existing layers.

```text
TO_PICKUP -> TO_DROPOFF -> next task
     |             |
     +--- VOLUNTARY_RETURN / FORCED_RETURN -> CHARGING -> resume task
```

Only collision, energy depletion, velocity violation, unrecoverable certificate failure, or the
configurable long-horizon limit ends a persistent episode. Persistent configs set
`terminate_on_terminal=false`; the default remains `true`, preserving single-mission behavior.

## Charging and energy authority

The first synthetic baseline uses capacity `30.0`, net charging rate `2.0` energy units/s, and
`dt=0.2` s, hence a gain of `0.4` per charging step. When and only when terminal position,
velocity, minimum energy, station availability, and evidenced hover continuation are valid,

\[
e_{t+1}=\min(e_{\max},e_t+r_c\Delta t).
\]

Charging uses `charger_hold`, does not invoke the task actor, and does not reset or teleport state.
A moving or non-docked UAV cannot charge.

The recovery certificate retains its first-passage meaning:

\[
m_E(z)=e-E^\kappa(z)-e_G.
\]

Future charging is not subtracted from current return cost. Reaching the forced-return margin
overrides service. Departure is accepted only with a valid persistent manifest and

\[
e\ge E_{\rm depart}^{\rm required}+m_e.
\]

Otherwise the UAV remains charging with `INSUFFICIENT_DEPARTURE_ENERGY`.

## Event-level scheduler

`reserve_only`, `fixed_threshold_30`, `fixed_threshold_50`, `full_charge`, and `scheduler_sac`
share task stream, plant, charging model, service manifest, Generator, and kappa. Decisions occur at
task boundaries and charging checkpoints, not every 0.2 s. The discrete scheduler is separate from
the affine-tanh physical actor and never uses Generator log density.

Scheduler replay stores requested/executed decisions, override reason, cumulative reward, duration,
scenario ID, and manifest hash. Its bootstrap multiplier is `gamma ** duration_steps`.
Scenario/manifest mismatches are rejected. Reward is based on persistent task efficiency and is not
a certificate source.

## Persistent certificate gate

`PersistentCertificateProvider` builds an edge-level mission certificate for every declared service
edge and composes their hashes. The gate requires complete task support, corridor-wide kappa chains,
docking admissibility, energy/E3 validity, departure data, switching closure, and version
consistency. Kappa and Generator certification remain separate. Charging creates no certificate.

The gate must be run manually before experiments; this round did not run it.

```bash
.venv/bin/python scripts/validate_persistent_certificate.py
.venv/bin/python scripts/run_persistent_env_acceptance.py --scenario persistent_open
.venv/bin/python scripts/run_persistent_scheduler_baselines.py
.venv/bin/python scripts/train_persistent_scheduler_sac.py --scenario persistent_energy_tight --steps 50000
.venv/bin/python scripts/evaluate_persistent_scheduler.py --scenario persistent_energy_tight --checkpoint <path>
```

These commands produce synthetic empirical evidence only.
