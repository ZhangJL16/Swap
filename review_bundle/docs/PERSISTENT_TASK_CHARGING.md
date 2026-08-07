# Persistent Goal Stream and Autonomous Charging

## Status and scope

This path is a single-UAV implementation scaffold with deterministic unit coverage. It contains no
fleet, order, or pickup/dropoff allocation layer. The persistent certificate gate,
acceptance rollout, energy-management baselines, learning, and evaluation scripts were not run in
this development round. All charging and physical-bound values are synthetic; no real-flight,
physical-calibration, HIL, or WCET claim follows.

The single-mission fixtures remain unchanged. Persistent fixtures are isolated as
`persistent_open`, `persistent_obstacle`, and `persistent_energy_tight`.

## Division of responsibility

```text
Environment                 assigns the next certified goal
Low-level policy            flies toward that fixed current goal
EnergyManagementPolicy      requests voluntary return or charger departure
Certificate runtime         verifies actions, forces recovery, and gates departure
```

The policy never selects a destination. `PersistentGoalTaskManager` samples a new node from the
finite `CertifiedGoalNetwork` after the current goal is reached. The charging station `S` is
excluded from `goal_node_ids`, so it cannot appear as a normal task goal.

The network distinguishes three route types:

- `TASK_EDGE`: service-node transit only; neither endpoint is `S`;
- `RECOVERY_EDGE`: a certified route from a service node to `S`;
- `DEPARTURE_EDGE`: a certified route from `S` to a pending goal.

The persistent state machine is:

```text
TASK -> next goal -> TASK -> ...
  |                         ^
  +-> VOLUNTARY_RETURN -----|
  +-> FORCED_RETURN -> CHARGING -> verified departure
```

A charging interruption marks the current `PersistentGoalTask` as interrupted but preserves its
task ID and goal. Leaving the charger plans a `DEPARTURE_EDGE` route to that same pending goal. A
new goal is assigned only after the pending goal is actually reached. Goal completion, station
arrival, and charging do not terminate the persistent episode.

## Charging and hard energy authority

The synthetic baseline uses capacity `30.0`, net rate `2.0` energy units/s, and `dt=0.2` s, hence
`0.4` energy units per admissible charging step:

\[
e_{t+1}=\min(e_{\max},e_t+r_c\Delta t).
\]

Charging requires terminal position, terminal velocity, minimum energy, station availability, and
evidenced hover continuation. It uses `charger_hold`, never teleports the UAV, and never resets
position, velocity, or battery. A moving or non-docked UAV cannot charge.

The frozen recovery certificate keeps its first-passage meaning:

\[
m_E(z)=e-E^\kappa(z)-e_G.
\]

Approaching the configured margin forces `FORCED_RETURN` regardless of the requested energy
decision. Future charging is never subtracted from the energy needed to reach the station.
Departure is allowed only when a version-matched manifest proves a route to the pending goal plus
its certified return reserve:

\[
e\ge E_{S\rightarrow g}^{\rm depart}+E_{g\rightarrow S}^{\kappa}+e_G+m_e.
\]

Otherwise `SERVE_OR_LEAVE` is overridden by `INSUFFICIENT_DEPARTURE_ENERGY` and charging continues.

## Energy management SMDP

`EnergyDecision` has two context-dependent actions:

- flight: `SERVE_OR_LEAVE` continues the fixed goal; `CHARGE_OR_STAY` requests voluntary return;
- charging: `CHARGE_OR_STAY` continues charging; `SERVE_OR_LEAVE` requests departure.

Decisions occur at new-goal/task-completion events and charging checkpoints. The safety runtime may
force return at any flight step. `EnergyManagementSAC` is an independent categorical SMDP learner;
it is not part of the Generator affine-tanh density. Replay stores requested/executed decisions,
override reason, accumulated reward, duration, scenario ID, and manifest hash, with bootstrap
factor `gamma ** duration_steps`.

Available energy-management policies are `reserve_only`, `fixed_threshold_30`,
`fixed_threshold_50`, `full_charge`, and `energy_management_sac`. All share the same goal stream,
plant, charging model, Generator, kappa, and certificate manifest. Legacy scheduler class/script
names are deprecated aliases only.

## Persistent certificate gate

`PersistentGoalCertificateProvider` composes typed edge manifests. `PERSISTENT_CERTIFICATE_GATE`
requires:

1. every pair of service goals has a certified task route;
2. every service goal has a certified recovery route to `S`;
3. `S` has a certified departure route to every pending goal;
4. complete successor support and corridor-wide kappa recovery remain valid;
5. energy recursion/E3, terminal docking, hashes, and versions are consistent;
6. interruption/resume changes route semantics but never forges a certificate.

Kappa and Generator certificates remain distinct. Charging does not reduce the return-energy bound
and creates no physical certificate.

## Manual experiment commands

These scripts were created but intentionally not run in this round:

```bash
.venv/bin/python scripts/validate_persistent_certificate.py
.venv/bin/python scripts/run_persistent_env_acceptance.py --scenario persistent_open
.venv/bin/python scripts/run_energy_management_baselines.py
.venv/bin/python scripts/train_energy_management_sac.py --scenario persistent_energy_tight --steps 50000
.venv/bin/python scripts/evaluate_energy_management.py --scenario persistent_energy_tight --checkpoint <path>
```

Any output is synthetic empirical evidence only.
