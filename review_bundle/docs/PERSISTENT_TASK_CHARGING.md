# Persistent Goal Stream with Certified Recoverability Backup

## Status and scope

The main persistent method has one trainable policy: `PersistentGeneratorSAC`. It emits only a
three-dimensional continuous latent. `EnergyManagementSAC` and its categorical SMDP remain
ablation-only compatibility code and are never instantiated by `make_persistent_uav_env`.
Persistent certificate validation, acceptance rollouts, training, and evaluation were deliberately
not run in this implementation round. All bounds and charging values remain synthetic.

## Authority and task semantics

```text
Environment             assigns the next certified goal
PersistentGeneratorSAC  controls task flight and voluntary station approach
Certificate runtime     restricts every accepted action to A_rec
Frozen kappa            acts only after backup takeover
Charger support         constrains the complete Generator to remain docked while departure is closed
Charger hold            exceptional fallback if constrained support cannot be certified
```

The environment samples goals from the finite `CertifiedGoalNetwork`; the charging station is not a
normal goal. Reaching a goal assigns the next goal without terminating the episode. A voluntary
station visit preserves the task ID and pending goal. Leaving the charger resumes that same goal.

The normal action is

\[
u_t\sim\pi_\theta(\cdot\mid o_t),\quad \eta_t=\tanh u_t,\quad
a_t=c(z_t)+G(z_t)\eta_t,
\]

where `u_t` has dimension three. The persistent default center is `safety_neutral`; it does not
encode the task-goal direction or station-return decision. The complete Generator must remain
full-rank and must pass the recoverability verifier.

## Recoverability certificates

The frozen recovery-energy field retains its undiscounted robust first-passage meaning. Define

\[
\mathcal R=\{z:\text{a valid certified kappa chain exists and }
e-E^\kappa(z)-e_G-m_e\ge 0\}.
\]

Membership means kappa is available as a certified backup; it does not mean kappa currently has
control. The admissible learned-action authority is

\[
\mathcal A_{\rm rec}(z)=\{a:\operatorname{Post}(z,a)\subseteq\mathcal R\}.
\]

`Post` is the uncertainty-aware interval/zonotope successor envelope. The verifier jointly checks
actuator limits, velocity bounds, swept FREE geometry, tracking/dynamics bounds, and the successor
energy inequality

\[
e^+_{\rm lower}\ge E^\kappa(z^+)_{\rm upper}+e_G+m_e.
\]

The state-level Generator is accepted only after complete-set verification:

\[
c(z)+G(z)[-1,1]^3\subseteq\mathcal A_{\rm rec}(z).
\]

No sampled action, center, or finite rollout substitutes for this inclusion check.

## T_REC1 and T_REC2

**T_REC1 (one-step recoverability preservation).** If `z_t` belongs to `R` and runtime publishes an
action from a verified `C_run(z_t) subset A_rec(z_t)`, every state represented by the certified
successor envelope belongs to `R`.

**T_REC2 (recursive recoverability).** If `z_0` belongs to `R` and every learned action is published
from a newly verified `C_run`, induction on T_REC1 gives `z_t in R` for every normal-policy step.
This preserves the existence of a certified recovery option; it does not claim that the learned
policy itself returns to the station.

At the configured interior switching margin, `NO_GENERATOR_SET`, invalid evidence/version,
watchdog failure, or another task-certificate failure, authority switches to kappa. The existing
strict corridor descent and E3 energy recursion then provide the conditional finite-time return
result. If kappa's own certificate is invalid, execution fails closed.

## Voluntary charging and departure

Voluntary station approach is inferred from continuous behavior, not a discrete policy output. If
the UAV reaches the charging admissible set without backup takeover, RL authority remains active and
the visit is logged as voluntary. Remaining inside the set at admissible velocity applies synthetic
net charging

\[
e_{t+1}=\min(e_{\max},e_t+r_c\Delta t).
\]

The fixture uses capacity `30.0`, rate `2.0` units/s, and `dt=0.2` s (`0.4` units per cycle).
These are not calibrated physical values.

While the departure gate is closed, the certificate path constructs
`C_charge(z) subset A_rec(z) intersection A_stay(z)`, where every successor remains in the charging
set. The normal accepted policy action is therefore also the physically executed action; unsafe
departure directions are absent from its support rather than post-hoc replaced. When the departure
energy and manifest checks pass, the ordinary `C_run subset A_rec` support is restored. Certified
zero hold remains only an explicit certificate/numerical fallback. The pending goal is unchanged,
and future charging never reduces the energy required to reach the station from a flight state.

## Manifest and policy-authority gates

The aggregate manifest stores one `SharedBoundVersions` object for dynamics, tracking, energy,
terminal, recoverable-set/action rules, and the runtime configuration. Those assumptions must match
across every edge. Geometry, corridor, mission-manifest, and kappa identities are edge-local; each is
hash-bound by an `EdgeDependencyBinding`, and every binding hash enters the aggregate manifest.
Different edge-local IDs are therefore expected, while a changed shared bound or a tampered edge
dependency remains a hard `VERSION_MISMATCH`.

`PERSISTENT_SAFETY_GATE` applies typed prerequisites. `TASK_EDGE` and `DEPARTURE_EDGE` require their
recovery chain plus verified task/departure successor support. `RECOVERY_EDGE` requires only the
complete kappa chain, strict descent, geometry, actuator/velocity bounds, E3, terminal linkage, and
hashes; it does not require a full-rank Generator. A recoverable state with `NO_GENERATOR_SET` is a
valid kappa-backup state, not automatically a safety-certificate failure.

`POLICY_AUTHORITY_GATE` checks only states where normal RL authority is represented: task,
departure, and constrained charging support. Task states still require both goal- and
station-directed residual authority. Recovery-only cells are audited separately for kappa validity.
`POLICY_AUTHORITY_COVERAGE` reports how many eligible RL roots have a verified full-rank Generator;
coverage is a learnability/performance metric, not a safety theorem. Sigma, condition, and volume
aggregates exclude kappa-only cells. These remain synthetic software gates, not physical calibration.

The latest corrected synthetic validation passes both gates for `persistent_open` and
`persistent_energy_tight` with 1353/1353 RL-authority roots and all 36984 kappa-only cells valid in
each scenario. `persistent_obstacle` fails the safety and policy gates: 906 `recover_C_S`, 591
`task_C_B`, 1017 `task_C_D`, and 809 `task_D_C` cells have complete swept-geometry containment
failures (`minimum_geometry_slack=-1.0` in the first witnesses). Their hashes, E3 residuals,
velocity bounds, and strict descent links remain valid. This is a real synthetic certificate
infeasibility, not a version bug, typed-gate bug, or `NO_GENERATOR_SET` constructor failure.

## Replay and optimization

Replay records `u`, `eta`, `c`, `G`, candidate, executed and measured actions, backup state/reason,
energy margin, charging/departure events, pending task, manifest, and bound versions. Critics train
on `executed_action`. Only accepted Generator transitions use the affine-tanh continuous density;
backup atoms do not. `c` and `G` remain detached during actor updates.

`PersistentExecutionAuthority` is the single immutable classifier used to serialize runtime
authority into replay. Its outcomes are `RL_GENERATOR`, `KAPPA_BACKUP`, `CHARGER_CONSTRAINED`, and
`FAIL_CLOSED`. `PersistentGeneratorSAC` selects its next-state Bellman branch from that recorded
authority, not merely from mathematical Generator existence. A mandatory next-step backup uses
`kappa(z_next)` without Generator entropy; a closed charger uses the certified `C_charge` Generator
when available, otherwise an explicitly recorded atomic hold; fail-closed next states do not
bootstrap. This is runtime/training semantic closure, not a new SAC convergence claim.

## Manual commands

The following formal commands were created or updated but were not run in this round:

```bash
.venv/bin/python scripts/validate_persistent_certificate.py
.venv/bin/python scripts/run_persistent_env_acceptance.py --scenario persistent_open --probe all --strict
.venv/bin/python scripts/train_persistent_generator_sac.py --scenario persistent_open --legacy-fixed-graph --steps 50000
.venv/bin/python scripts/evaluate_persistent_generator_sac.py --scenario persistent_open --legacy-fixed-graph --checkpoint <path>
.venv/bin/python scripts/run_persistent_single_policy_baselines.py --scenario persistent_open --legacy-fixed-graph
```

The old `train_energy_management_sac.py` and related scripts are explicitly hierarchical ablations,
not the main method. Any future outputs remain synthetic empirical evidence and cannot establish
real-flight safety, calibrated physical bounds, or hard WCET.

## Task-independent random-goal main problem

The main path separates physical/certificate state `x` from externally assigned task goal `g`.
The actor is goal-conditioned, `pi_theta(a | x, g)`, but the certified support is not:

```text
A_safe(x) = A_act(x) intersect A_col(x) intersect A_rec(x)
C_run(x) = c(x) + G(x)[-1,1]^3 subset A_safe(x)
```

`CertifiedRecoverabilityAtlas` covers a certified subset of the free workspace with recovery
cells. Each cell binds geometry, dynamics, tracking, energy, terminal, and frozen-kappa proof
dependencies. Atlas construction consumes no current goal, goal seed, task edge, task waypoint,
route index, or reward. `RandomPersistentTaskWrapper` samples reproducible starts and continuous
horizontal goals only from certified atlas interiors. Reaching a goal samples the next goal
without resetting the plant; charging and backup preserve the same pending goal.

The fixed `CertifiedGoalNetwork` and `TASK_EDGE` fixtures remain legacy regressions and ablations.
They are not prerequisites for normal authority in the random-goal main method.

## T_RAND contracts

**T_RAND1 (random certified initialization).** If the initial distribution has support inside the
certified atlas, the existing T_REC initialization premise holds.

**T_RAND2 (goal-independent recursive recoverability).** For any admissible goal sequence and any
goal-conditioned learned policy, T_REC2 remains valid when every normal action is published from
the task-independent `C_run(x) subset A_rec(x)`. This guarantees recoverability, not sampled-goal
completion.

**T_RAND3 (goal-independent support).** At identical physical/certificate state and versions,
changing only `g` leaves `E^kappa`, recoverable membership, kappa proof, `c`, `G`, action bounds,
and atlas identity unchanged. Actor output may change.

## Recovery versus RL-authority viability

`R` contains every state with a certified finite kappa return and sufficient recovery energy.
`R_RL` is the task-independent atlas fixed point of recoverable cells that also have full-rank
Generator support with a complete successor in `R_RL` or `G_charge`.  Therefore
`R_RL subset R`; cells in `R` but outside `R_RL` remain legitimate kappa-only recovery cells.

Normal support additionally satisfies `C_run subset A_cont`, where `A_cont` preserves `R_RL` or
enters the certified charging set.  The safety-neutral center may apply atlas-state feedback needed
for this invariant support, but it receives no goal, task route, waypoint, or reward.  Goal changes
must leave `R`, `R_RL`, `E^kappa`, `c`, `G`, continuation target, and certificate identity unchanged.

The charging terminal has a formal level-zero recovery certificate.  It binds terminal geometry,
dynamics, tracking, energy, terminal and kappa versions, and the atlas core hash; its recovery
energy upper bound is zero. It additionally binds a local terminal hold controller whose complete
successor envelope must stay in the charging set; zero-step recovery therefore does not imply a
zero acceleration command in the presence of residual velocity. This prevents completed recovery
from being reinterpreted as a missing nonterminal successor. A closed departure gate uses charger-constrained support or certified hold;
an open departure is accepted only when its complete successor returns to `R_RL`.

Generator-SAC diagnostics decompose normalized and physical log density. The physical density keeps
the affine determinant term exactly. The controlled normalized-temperature candidate changes only
the automatic alpha residual to use `log pi_eta`; actor and Bellman terms still use `log pi_a`, so
the executed policy density and certificate semantics are unchanged. This is equivalent to a
state-dependent physical entropy target shifted by `log|det G|` and is invariant to uniform affine
support scaling.

The persistent reward uses `backup_intervention_cost` as an event cost only when authority first
transfers into `KAPPA_BACKUP`. Kappa continuation does not repeat that intervention charge. Recovery
steps still pay elapsed-time and energy costs, and charging steps retain their dwell cost. Telemetry
therefore distinguishes `backup_recovery_count`, `kappa_backup_steps`, and
`backup_intervention_reward_events`.

## Task-neutral support expressiveness

Random-persistent normal support no longer follows the directed velocity/action of the offline
coverage trace. Coverage positions seed zero-velocity safety cells; each cell retains its own
independently certified kappa return chain. The normal center is a local position/velocity
stabilizer, while multiple goal-independent viable successor cells are considered and Generator
scales are enlarged only through the complete verifier. Task goals affect the policy latent, not
`c`, `G`, recovery evidence, or successor eligibility.

The best-in-Generator oracle is a diagnostic, not a controller in the main method and not a safety
proof. It establishes whether a goal-aware selector can make progress inside the certified support
before additional SAC training is justified.
