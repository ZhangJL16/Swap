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

The persistent manifest binds the recoverable-set, recoverability-action-rule, energy-field, kappa,
geometry, tracking, and dynamics versions. `PERSISTENT_CERTIFICATE_GATE` requires route closure,
strict recovery descent, E3, terminal linkage, recoverable state cells, complete Generator
recoverability, departure recoverability, and hash/version consistency.

`POLICY_AUTHORITY_GATE` checks every persistent edge and representative certified root rather than
only the reset point. It reports output dimension, neutral center, full rank, minimum sigma, maximum
condition number, minimum volume, task/station directional authority, complete-set recoverability,
and exact failed locations. This is a software gate, not a physical calibration result.

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
.venv/bin/python scripts/train_persistent_generator_sac.py --scenario persistent_open --steps 50000
.venv/bin/python scripts/evaluate_persistent_generator_sac.py --scenario persistent_open --checkpoint <path>
.venv/bin/python scripts/run_persistent_single_policy_baselines.py --scenario persistent_open
```

The old `train_energy_management_sac.py` and related scripts are explicitly hierarchical ablations,
not the main method. Any future outputs remain synthetic empirical evidence and cannot establish
real-flight safety, calibrated physical bounds, or hard WCET.
