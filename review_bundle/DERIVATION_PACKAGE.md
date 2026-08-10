# Derivation Package

## Target

Determine whether a mathematically coherent unified constrained reinforcement-learning method can be built for persistent UAV missions with:

- collision avoidance;
- preservation of enough energy to return to a charging station;
- a task policy trained by online off-policy maximum-entropy RL;
- takeover by a frozen certified recovery policy;
- explicit rolling local geometry and a sparse certified return corridor, with learned collision and energy fields used only as proposal functions;
- a feedforward CNN/MLP task pathway separated from a non-neural geometric certification pathway;
- strict guarantees restricted to a continuously verified recovery corridor.

This package fixes Route A as the primary task-RL route, Route C as fallback, and Route B as auxiliary regularization/ablation. The runtime realization is now fixed to a full-rank three-generator affine zonotope driven by a tanh-squashed three-dimensional actor. The strict certificate state contains no recurrent neural hidden state. Collision-field semantics, uncertainty semantics, neighborhood verification geometry, and the recovery-policy replacement rule remain open.

## Status

**COHERENT AFTER REFRAMING / EXTRA ASSUMPTION**

The target is theoretically feasible only after the following reframe:

1. A learned field is a proposal or approximation, not a certificate.
2. The strict theorem is attached to a verified lower/upper envelope over a certified corridor, not to the neural output over the whole state space.
3. The frozen recovery policy and corridor establish recursive feasibility and finite-time return; the task RL algorithm may exploit this structure but cannot create it by critic regularization alone.
4. The energy field is an undiscounted first-passage cost under the frozen recovery policy. Its uniqueness must come from properness, finite-level descent, or a weighted-norm argument—not from the task discount factor.
5. Any maximum-entropy policy theorem is stated for a frozen, measurable, nonempty certified action correspondence. When that correspondence changes during learning, the standard soft policy-improvement proof no longer compares the same control problem.
6. The certificate source and the action-enforcement mechanism are separate objects. A verified action set certifies which actions are safe; a mask or projection can only enforce membership in that set. A learned editor without a verified range-inclusion property supplies empirical violation reduction, not certification.
7. The strict state is \(z_t^{\rm cert}=(p_t,v_t,e_t,p_{\mathcal G},\mathcal M_t^{\rm local},\mathcal C_t^{\rm back},\xi_t)\). Its geometric components are explicit finite representations with verified update rules; no GRU state is propagated, bounded, or invoked in a physical theorem.
8. Route A is realized by \(a=c(z)+G(z)\tanh u\), where \(G(z)\in\mathbb R^{3\times3}\), \(\sigma_{\min}(G(z))\ge\sigma_G>0\), and the complete zonotope \(c(z)+G(z)[-1,1]^3\) is continuously verified as an inner subset of \(\mathcal A_{\rm cert}(z)\).
9. The executable reference profile represents \(\mathcal M^{\rm local}\) as a finite world-aligned ternary grid and \(\mathcal C^{\rm back}\) as a finite station-to-UAV chain of overlapping AABBs. It uses interval successor propagation and deterministic diagonal three-generator inner-zonotope construction. These are conservative implementation choices, not claims that the required uncertainty bounds have already been calibrated on the real aircraft.

The key feasibility verdict is therefore:

> A rigorous paper is possible if the neural fields are separated from their certified envelopes and if the recovery corridor is verified independently of the task critic. A theorem claiming strict safety from field-training losses, finite rollouts, or conservative Q-values alone is not presently feasible.

The strongest current conclusion is a **corridor-conditional robust guarantee**: all physical guarantees are conditional on starting inside the continuously verified corridor and on the verified envelopes, recovery policy, runtime verifier, and uncertainty set remaining valid.

## Complete Change Log

| Location | Original problem | Revision |
|---|---|---|
| A9b, D11--D12, runtime rule, M4, T0 | Fallback \(\kappa(z)\) was certified only in \(\mathcal A_{\rm cert}(z)\), but T0 incorrectly placed every outcome in \(C_{\rm run}(z)\) | Separate successful candidate membership in \(C_{\rm run}\) from overall execution membership in \(\mathcal A_{\rm cert}\); introduce nonconvex \(C_{\rm safe\text{-}run}=C_{\rm run}\cup\{\kappa\}\) only as an execution-range notation |
| D6 and energy margin | Terminal set omitted minimum terminal energy and charging admissibility | Add \(e_G\), \(V_G\), and \(\operatorname{ChargeAdmissible}\); reserve transit energy separately from terminal energy |
| A8, D8, L6, T3--T5 | One-step and bounded-\(M\)-step descent were merged into one backward-induction claim | Split one-step induction (T4a) from a stopped \(M\)-step block operator (T4b) |
| Route B | Fixed-point formula used an unrendered coefficient name | Replace it with the defined coefficient \(\lambda\) |
| Route A, T8A--T9A | Pointwise measurability and operator domain were underspecified | Add measurable-graph, reference-measure, partition-function, stochastic-kernel, and bounded-function-space conditions |
| T1, T6, T7 | Certified initial conditions and switching latency premises were implicit | State initial set membership, trigger timing, fallback, and delay-enclosure premises explicitly |
| Dependency graph | The energy proof and recovery-action certification risked circular presentation | Order progress \(\to\) energy recursion \(\to\) energy upper bound \(\to\) \(\kappa\)-certification \(\to\) task-action certification \(\to\) runtime enforcement |
| Certificate-state architecture | Strict claims depended on an unverified 128-dimensional recurrent state | Remove the GRU from the certificate and define \(z^{\rm cert}=(p,v,e,p_{\mathcal G},\mathcal M^{\rm local},\mathcal C^{\rm back},\xi)\) |
| Perception architecture | Task perception and strict certification shared an implicit neural context | Split the system into a feedforward task-perception path and an explicit geometric certification path |
| Local geometry | Unknown space and historical geometry were not represented as theorem-bearing objects | Define verified-free, verified-obstacle, and unknown sets; prohibit corridor/action certification through unknown space |
| Fields and recovery policy | Collision field, energy field, and \(\kappa\) accepted the recurrent context | Remove that input; fields remain proposals and \(\kappa\) depends only on the explicit certificate state |
| Route-A realization | Four enforcement mechanisms remained conditional | Select the full-rank three-generator zonotope \(C_{\rm run}=c+G[-1,1]^3\); retain exact truncation and projection only as controls |
| T12A | The theorem treated a generic diffeomorphic map | Specialize it to \(u\mapsto c+G\tanh u\) and derive its density, tanh and affine Jacobians, entropy, and exact SAC actor gradient |
| Runtime failure | Candidate failure was represented abstractly | Reject timeout, invalid inclusion, numerical failure, and \(\sigma_{\min}(G)<\sigma_G\), then execute certified \(\kappa\) |
| Remaining gaps | GRU certification was listed as the main bottleneck | Replace it with rolling-geometry verification, return-corridor maintenance, sound real-time zonotope construction, regularity of \(c,G\), and acceptance-law consistency |
| Generator nonexistence | Recovery-action nonemptiness was at risk of being conflated with positive-volume task-set existence | Introduce \(\mathcal Z_G\); outside it \(C_{\rm run}=\varnothing\) and runtime directly executes \(\kappa\) |
| Hybrid critic domain | T14A evaluated a pure-\(C\) critic at \(\kappa(z)\), which may lie outside \(C(z)\) | Introduce the augmented graph \(\mathsf G_{\rm fs}\) and bounded hybrid critic \(Q_{\rm fs}\) |
| Executable certificate state | \(\mathcal M^{\rm local}\) and \(\mathcal C^{\rm back}\) were abstract finite sets | Select a fixed-resolution ternary grid and a finite overlapping AABB corridor chain with explicit versions and evidence provenance |
| LiDAR update | L0 assumed a sound perception update without an algorithm | Add full-cell beam-tube free certification, inflated obstacle-hit updates, stale-cell expiry, and unknown-by-default semantics |
| Successor envelope | A1 did not specify a runtime set propagator | Add affine interval propagation covering state error, dynamics disturbance, action tracking, energy error, and versioned map/corridor updates |
| Recovery controller | \(\kappa\) was a theorem object without executable structure | Add deterministic corridor tracking, speed limiting, braking, and a one-step progress verifier using only the certificate state |
| Inner zonotope | L5a assumed a verified full-rank inner set without a constructor | Add lexicographic coordinate bisection around \(\kappa(z)\), exact actuator-box checks, Gershgorin singular-value certification, and full interval successor verification |
| Runtime and replay | RS1 lacked an executable interface | Add state-level set acceptance, actor bypass on certificate failure, immediate \(\kappa\) fallback, and separate nominal/candidate/executed replay fields |
| Verification evidence | The dependency graph did not identify concrete code evidence | Add an A1--A15, L0--L7, and T0--T14A implementation/evidence/metric ledger with unresolved premises marked explicitly |
| L0 calibration closure | The grid update omitted attitude, beam width, timing, footprint, discretization, expiry provenance, historical-obstacle dominance, and an explicit calibration gate | Add validated `SensorCalibrationContract`, full-cell free-tube tests, chord-radius hit dilation, permanent static-obstacle dominance, boundary exclusion, evidence provenance and expiry; mark physical L0 `blocked-by-calibration` until bounds are supplied |
| L1 numerical closure | Successor propagation used ordinary floating-point endpoints and omitted initial-state intervals, wind, period error, and latency | Add an independent outward-rounded `Interval` algebra and propagate the complete action interval, state uncertainty, tracking, wind, dynamics, energy, latency, and version ranges |
| L4--L6 recovery authority | Runtime trusted Boolean recovery-validity flags and pointwise controller checks | Replace flags with hashed, versioned, expiring `RecoveryCellCertificate` objects generated by a complete-cell one-step verifier; uncertified cells cannot authorize task mode or certified fallback |
| T4a--T5 energy closure | Corridor cells stored unverified scalar energy values | Add outward-rounded level-order backward recursion, hashed `RecoveryEnergyCertificate` objects, E3 residual verification, and separate terminal energy (e_G) |
| L5a construction closure | Generator checks omitted conditioning and certificate provenance | Verify full action intervals, σ-minimum, condition number, actuator bounds, collision/corridor/energy predicates, proof hashes, versions, and deadlines; return `NO_GENERATOR_SET` without reducing σ_G |
| T7 deadline closure | Synchronous elapsed-time checks could not protect against a blocking certifier | Add an independent default-κ watchdog state machine, immutable snapshots, complete atomic candidate bundles, one-shot command publication, and a `WCETContract`; real-time closure remains `blocked-by-deployment-evidence` |
| T12A implementation closure | The actor density used an unstated numerical treatment and replay omitted certificate details | Use the stable softplus tanh-Jacobian identity, detach (c,G), train critics on (a_{\rm exec}), and record certificate snapshots, versions, hashes, actions, acceptance, and fallback reasons |
| Strict audit | Previous statuses upgraded continuous premises from unit tests or input flags | Replace the ledger with item-wise mathematical premise, code object, evidence, missing content, closure decision, and controlled status; tests remain software evidence rather than physical proofs |
| Calibration package | Physical bounds lacked common evidence/version/domain/confidence schemas | Add immutable sensor, tracking, dynamics, energy, and terminal contracts; split validation; evidence hashes; expiry/domain checks; deterministic-versus-statistical semantics; CSV loaders; and synthetic fixtures |
| Evidence identity | A contract version was not bound to exact evidence content | Add contract hashes, same-version/different-hash rejection, hash fingerprints in snapshots, and dependency invalidation rules |
| Fixed-corridor closure | Geometry, recovery, energy, and zonotope proofs were generated separately | Add `SingleCorridorClosurePipeline`, all-cell fail-closed behavior, interval failure witnesses, linked proof metadata, and an integrity-checked manifest |
| WCET evidence | Deadline logic lacked reproducible measurement infrastructure | Add warm-up, affinity interface, size sweeps, median/p99/maximum timing, and an explicit desktop-profiling versus hard-WCET status boundary |
| Closed-loop integration | Control sequencing and failure injection were not executable end to end | Add mock/replay HIL adapters, deterministic cycle orchestration, staged-κ watchdog traces, atomic publication, replay, audit records, and requested failure injections |
| Generator-SAC integration | T12A had no certificate-epoch trainer semantics | Add `GeneratorSACTrainer`, frozen `CertificateEpoch`, `a_exec` critic input, accepted-branch entropy only, fallback-atom exclusion, and replay-version rejection |
| Torch evidence | Formula/gradient tests were skipped in the reference interpreter | Run all actor/trainer tests in the UV-managed `.venv` (Torch 2.7.1+cu128) and the complete 92-test suite with zero skips and zero failures |
| Symbol audit | Legacy runtime parameter presence checks and physical evidence contracts shared `SensorCalibrationContract` | Rename the legacy object to `RuntimeSensorBoundsContract`; reserve `SensorCalibrationContract` for the evidence-bearing calibration object |

## Invariant Object

The common object across all three RL routes is the **verified one-step predecessor relation** induced by a frozen recovery corridor:

\[
(z,a)\in\operatorname{Pre}_{\rm cert}(\mathcal C)
\quad\Longleftrightarrow\quad
\operatorname{Post}(z,a)\subseteq \mathcal C
\text{ and every successor preserves collision and energy margins.}
\]

Here \(z\) is the information state, \(a\) is the actually executed acceleration, \(\operatorname{Post}(z,a)\) is a conservative set containing every real successor, and \(\mathcal C\) is the verified recovery corridor. The three routes differ in how this relation alters policy evaluation and policy improvement:

- Route A makes it the support of the task policy.
- Route B converts its margin or confidence into a critic penalty.
- Route C learns separate task and constraint values, then uses a feasibility test in policy improvement.

The safety theorem ultimately depends on membership in \(\operatorname{Pre}_{\rm cert}(\mathcal C)\), regardless of how the task policy is trained.

## Assumptions

The assumptions below are requirements, not facts already established.

| ID | Assumption required for a strict theorem |
|---|---|
| A1 | The real certificate-state successor is contained in an independently outward-rounded, versioned set envelope: \(z_{t+1}^{\rm cert}\in\operatorname{Post}_{\rm cert}(z_t^{\rm cert},a_t)\) for every admissible initial-state error, full action set, tracking error, wind/residual disturbance, control-cycle/latency error, energy error, local-geometry update, and return-corridor update. All bound versions used by a certificate match runtime. |
| A2 | A calibrated, versioned one-step energy contract satisfies \(0\le c_{\rm real}(z^{\rm cert},a,w)\le \bar c(z^{\rm cert},a)\) throughout every certified state/action cell; interval evaluation uses the full cell and full action set. |
| A3 | Collision certification is computed from explicit geometric sets. Any learned collision field is only a proposal; the theorem uses a verified lower envelope \(\underline B_{\rm geom}\) over the complete certified neighborhood and successor tube. |
| A4 | Any learned recovery-energy field is only a proposal. The theorem uses a corridor-wise verified upper function \(\overline R^\kappa\) or an independently computed upper bound satisfying the residual recursion. |
| A5 | Obstacles are static and vertically extruded. Unknown space is non-certifiable. A complete `SensorCalibrationContract` bounds pose, attitude/direction, range, beam width, synchronization, footprint, discretization, evidence age, and motion during delay. For every certified state, verified-free space contains the complete control-delay, reaction, braking, tracking-error, and estimation tube; otherwise no task action is certified. |
| A6 | The charging terminal set is collision-safe and is represented by a parameterized `TerminalCondition`: terminal position/altitude sets, velocity interval \(V_G\), minimum energy \(e_G\), and at least one verified hover/descent/docking continuation mode. |
| A7 | The frozen recovery policy \(\kappa(z^{\rm cert})\) uses only explicit certificate-state variables. Every corridor state cell has a linked, hashed, versioned, unexpired complete-cell certificate proving actuator bounds, corridor/stopping-tube containment, velocity bounds, and the declared successor relation. A Boolean validity input is not evidence. |
| A8 | The frozen recovery policy satisfies one explicitly declared progress premise. **A8-1:** every nonterminal successor obeys \(\ell(z^+)\le\ell(z)-1\). **A8-M:** for every admissible recovery trajectory starting from \(z\in K_i\setminus\mathcal G\), the stopping index \(\sigma_i:=\inf\{j\ge1:z_j\in\mathcal G\text{ or }\ell(z_j)<i\}\) satisfies \(1\le\sigma_i\le M\). T4a uses A8-1; T4b uses A8-M. |
| A9a | The measurable generator-enabled domain \(\mathcal Z_G\subseteq\mathcal S_{\rm joint}^{\rm cert}\) contains exactly the certified states where a valid full-rank set is available. For \(z\in\mathcal Z_G\), \(C_{\rm run}(z)=c(z)+G(z)[-1,1]^3\), \(G\in\mathbb R^{3\times3}\), \(\sigma_{\min}(G)\ge\sigma_G>0\), and a sound verifier establishes \(C_{\rm run}(z)\subseteq\mathcal A_{\rm cert}(z)\cap\mathcal A\). Outside \(\mathcal Z_G\), the task candidate branch is disabled. |
| A9b | An independent watchdog first snapshots the certificate state and stages already certified \(\kappa(z)\). A task candidate is published only after a complete atomic bundle with the matching version, \(c,G\), full-set inclusion result, finite actor sample, final action, and acceptance flag arrives within a verified WCET deadline. Every failure leaves \(\kappa\) published exactly once. |
| A9c | The critic and replay use the actually executed acceleration \(a_{\rm exec}\). Replay additionally stores \(u,\eta,c,G\), the acceptance indicator, and the fallback indicator. Actuator tracking error after \(a_{\rm exec}\) is included in \(\operatorname{Post}_{\rm cert}\); nominal and executed actions are never silently interchanged. |
| A10a | The explicit certificate-state space and compact actuator space are standard Borel spaces. The maps \(c:\mathcal Z_G\to\mathbb R^3\) and \(G:\mathcal Z_G\to\mathbb R^{3\times3}\) are Borel measurable, so \(C(z)=\operatorname{int}C_{\rm run}(z)\) has a measurable graph over \(\mathcal Z_G\). |
| A10b | The Route-A reference measure is three-dimensional Lebesgue measure. For \(z\in\mathcal Z_G\), \(\lambda_A(C(z))=8|\det G(z)|\ge8\sigma_G^3>0\); compact actuator bounds give a uniform upper bound. Hence the Route-A partition function is finite and positive for bounded \(Q\). No positive-volume claim is made outside \(\mathcal Z_G\). |
| A10c | Rewards are bounded measurable and \(P(\cdot|z,a)\) is a measurable stochastic kernel on the generator state-action graph. T8A's pure constrained operator additionally assumes this kernel is supported on \(\mathcal Z_G\); if a successor leaves \(\mathcal Z_G\), the full safeguarded process must use \(\kappa\) and is described by T14A rather than the pure T8A operator. |
| A10d | The trace-measurable \(Q\in\mathbb B_b(\mathsf G_C)\) admits its zero extension to the product space; hence \(\mathbf1_{\operatorname{Gr}(C)}e^{Q/\alpha}\) is jointly measurable. Parameter integration then makes \(z\mapsto Z_C^Q(z)\), \(V_C^Q(z)\), and \(\pi_C^Q(\cdot|z)\) measurable. |
| A11 | During each RL policy-evaluation/improvement proof, the fields, uncertainty envelopes, recovery policy, corridor, and certified action correspondence are frozen. |
| A12 | If guarantees are probabilistic rather than deterministic, every confidence statement is simultaneous over the claimed time horizon and certified cells; pointwise calibration is insufficient. |
| A13a | The feedforward actor outputs \(u=m_\theta(o^{\rm task})+s_\theta(o^{\rm task})\odot\varepsilon\in\mathbb R^3\), \(\varepsilon\sim\mathcal N(0,I_3)\), with positive finite standard deviations; \(\eta=\tanh u\) and \(a_M=c(z)+G(z)\eta\). |
| A13b | During one actor update, \(c(z)\), \(G(z)\), the verified corridor, and the critic are frozen. The Gaussian reparameterization is differentiable, differentiation may pass through the expectation, and all score, log-Jacobian, and critic-gradient terms in T12A are integrable. |
| A13c | The selected map \(u\mapsto c(z)+G(z)\tanh u\) is analyzed only for \(G\in\mathbb R^{3\times3}\) with \(\sigma_{\min}(G)\ge\sigma_G>0\). Generator counts \(g>3\), noninjective maps, rank loss, and dimension changes are not covered by the determinant theorem and revert to T10A's pushforward/induced-measure analysis. |
| A13d | Exact truncation T11A and projection T13A are retained only as theoretical/experimental controls. They are not the main runtime realization. |
| A13e | The fail-safe-mixture derivation assumes a measurable acceptance event with probability \(\beta_\theta(z)\), a well-defined conditional generator-candidate law, and a separately certified deterministic fallback \(\kappa(z)\). Differentiability statements exclude \(\beta_\theta\in\{0,1\}\). |
| A14 | \(\mathcal M^{\rm local}\) and \(\mathcal C^{\rm back}\) use finite explicit set representations with measurable, sound update correspondences. Verified-free, verified-obstacle, and unknown regions are disjoint; unknown space cannot be added to the corridor until reclassified as verified free. |
| A15 | The feedforward task observation \(o^{\rm task}=\Omega_{\rm task}(z^{\rm cert},y)\) contains current LiDAR, a finite encoding of the local geometry crop, the return-corridor encoding, and task variables. Strict certificates never depend on CNN/MLP features. T9A's exact policy-improvement claim additionally requires the actor class conditioned on this explicit input to realize the stated optimizer. |

## Notation

### Physical, certificate, and task states

- \(p=(p^{xy},p^z)\in\mathbb R^3\): absolute position.
- \(v=(v^{xy},v^z)\in\mathbb R^3\): velocity.
- \(e\in\mathbb R_{\ge0}\): remaining energy.
- \(p_{\mathcal G}\in\mathbb R^3\): broadcast charging-station position.
- \(a\in\mathcal A\subset\mathbb R^3\): actually executed acceleration.
- \(y\in\mathbb R^{32}\): current world-frame LiDAR distances and validity indicators.
- \(\xi\): explicit task variables needed by the task reward and transition model.

The rolling local geometry is

\[
\mathcal M^{\rm local}
=
(\mathcal F,\mathcal O,\mathcal U,\mathcal W),
\qquad
\mathcal W=\mathcal F\mathbin{\dot\cup}\mathcal O\mathbin{\dot\cup}\mathcal U,
\]

where \(\mathcal W\subset\mathbb R^2\) is the represented horizontal sensing window, \(\mathcal F\) is verified free space, \(\mathcal O\) is verified obstacle space, and \(\mathcal U\) is unknown space. The disjoint union is with respect to the conservative set representation; unresolved boundary cells belong to \(\mathcal U\).

The sparse return corridor is an ordered finite family

\[
\mathcal C^{\rm back}=(B_0,\ldots,B_L),
\]

where \(B_0\) is contained in the charging terminal neighborhood, consecutive cells overlap through verified transition gates, \(B_L\) contains the current certified neighborhood, and every \(B_i\) lies in verified free space with associated position, velocity, and energy envelopes. It is a return certificate, not a complete global SLAM map.

The strict certificate state is

\[
\boxed{
z^{\rm cert}
=(p,v,e,p_{\mathcal G},
\mathcal M^{\rm local},
\mathcal C^{\rm back},
\xi)
\in\mathcal Z_{\rm cert}.}
\]

Henceforth \(z\) abbreviates \(z^{\rm cert}\) unless a task observation or latent actor variable is explicitly named.

The feedforward task pathway receives

\[
o^{\rm task}
=\Omega_{\rm task}(z^{\rm cert},y),
\]

implemented by a CNN/MLP over current LiDAR, a finite local-geometry crop, the return-corridor encoding, and task variables. Neural features produced by this pathway are not components of \(z^{\rm cert}\) and are absent from every physical certificate.

### Dual-path architecture

1. **Task-perception path:** \(o^{\rm task}\mapsto(m_\theta,s_\theta)\mapsto u\mapsto\eta=\tanh u\). It proposes task actions and is trained by Route A.
2. **Certification path:** \(z^{\rm cert}\mapsto(\widehat{\operatorname{Post}}_{\rm cert},\underline B_{\rm geom},\overline R^\kappa,c,G,\kappa)\). It uses explicit geometry, bounded uncertainty, and continuous set verification. No CNN/MLP encoding is accepted as a proof premise.

The task path may fail without invalidating the certificate path; failure invokes \(\kappa\).

### Real dynamics and explicit geometry updates

Let \(w\in\mathcal W_{\rm unc}(z^{\rm cert},a)\) collect physical disturbance, tracking error, state-estimation error, sensor error, and timing uncertainty. Let \(\upsilon_M,\upsilon_C\) index every update consistent with the verified geometry and corridor-update correspondences. The real certificate-state evolution is

\[
\begin{aligned}
(p^+,v^+) &= f_{\rm kin}(p,v,a,w),\\
e^+ &= e-c_{\rm real}(z^{\rm cert},a,w),\\
\mathcal M^{{\rm local},+}
&\in\operatorname{Upd}_M(
\mathcal M^{\rm local},y^+,p^+,w;\upsilon_M),\\
\mathcal C^{{\rm back},+}
&\in\operatorname{Upd}_C(
\mathcal C^{\rm back},
\mathcal M^{{\rm local},+},
p^+,v^+,w;\upsilon_C),\\
\xi^+&\in\operatorname{Upd}_\xi(\xi,p^+,v^+,a,w).
\end{aligned}
\tag{S1}
\]

The learned residual dynamics model may propose an envelope, but the strict theorem uses only a verified outer envelope containing (S1).

### Successor set

\[
\operatorname{Post}_{\rm cert}(z,a)
:=
\left\{
z^+:
\begin{array}{l}
z^+\text{ satisfies (S1) for admissible }
w,\upsilon_M,\upsilon_C
\end{array}
\right\}.
\tag{S2}
\]

The computed envelope must satisfy

\[
\operatorname{Post}_{\rm cert}(z,a)
\subseteq
\widehat{\operatorname{Post}}_{\rm cert}(z,a).
\tag{S3}
\]

All later uses of \(\operatorname{Post}\) refer to \(\operatorname{Post}_{\rm cert}\). The envelope includes plant tracking after the final executed action, sensing/update delay, and all admissible changes to \(\mathcal M^{\rm local}\) and \(\mathcal C^{\rm back}\).

### Perception, braking, and unknown-space condition

Let \(\Delta_{\rm ctl}\) be the control period, \(\tau_{\rm lat}\) the certified worst-case perception/compute/switch latency, \(b_{\min}>0\) the guaranteed horizontal braking deceleration, and \(\varepsilon_{\rm geom}\) the combined footprint, estimation, tracking, and set-approximation margin. Define

\[
d_{\rm stop}(v)
=\|v^{xy}\|(\Delta_{\rm ctl}+\tau_{\rm lat})
+\frac{\|v^{xy}\|^2}{2b_{\min}}
+\varepsilon_{\rm geom}.
\tag{S4}
\]

A state is geometrically certifiable only when the complete robust delay-and-braking swept tube \(\operatorname{Tube}_{\rm stop}(z)\) lies in \(\mathcal F\), equivalently when no part of that tube intersects \(\mathcal O\cup\mathcal U\). A scalar sensing-range check \(r_{\rm sens}\ge d_{\rm stop}(v)\) is necessary but not sufficient; directional free-space containment is also required.

### Field semantics without neural hidden state

The collision proposal field has the form

\[
B_\theta(
p_q^{xy},v_q^{xy},
\operatorname{Enc}(\mathcal M^{\rm local},
\mathcal C^{\rm back})),
\]

and the energy proposal field has the form

\[
E_\psi(
p_q,v_q,p_{\mathcal G},
\operatorname{Enc}(\mathcal M^{\rm local},
\mathcal C^{\rm back}))\ge0.
\]

Neither field receives remaining energy as an input to the recovery-cost estimate, and neither receives a recurrent hidden state. The collision theorem uses a verified geometric lower envelope \(\underline B_{\rm geom}\), obtained from explicit free/obstacle/unknown sets and successor tubes. The energy theorem uses \(\overline R^\kappa\), verified from the explicit corridor, energy-cost envelope, and frozen \(\kappa\).

For the later-selected risk functional \(\rho\),

\[
R_\rho^\kappa(z^{\rm cert})
=
\rho\!\left(
\sum_{t=0}^{\tau_{\mathcal G}-1}
c_{\rm real}(z_t^{\rm cert},
\kappa(z_t^{\rm cert}),w_t)
\right),
\]

where

\[
\bar z^{\rm cert}
=(p,v,p_{\mathcal G},
\mathcal M^{\rm local},
\mathcal C^{\rm back},
\xi)
\]

excludes remaining energy. The true first-passage cost remains a function of the full certificate state because \(\kappa\), admissible dynamics, and stopping can depend on \(e\). The learned field instead proposes a single upper value for each reduced-state fiber, and strict results require the independently verified envelope

\[
\overline R^\kappa(\bar z^{\rm cert})
\ge
\sup\left\{
R_\rho^\kappa(z^{\rm cert}):
z^{\rm cert}\in\mathcal C,
\operatorname{drop}_e(z^{\rm cert})=\bar z^{\rm cert}
\right\}.
\]

Thus omitting remaining energy from the field is a uniformity obligation, not an assertion that the true recovery cost is energy-independent.

## Literature Matrix

The matrix uses Guided Flow Policy, Flow Actor-Critic, Stolz et al. (2024), Yu et al. (2022), and Markgraf et al. (2026). The local OSR PDF used during drafting was malformed, so theorem extraction used the [official NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7a0f7e9d9b42b26e5bfc9ba4c6e5287c-Abstract-Conference.html).

| Source | Setting and model | Constraint/safety object | Starting optimization and operator | Main assumptions and results | Proof tools and guarantee type | Transferable to the UAV method | Not transferable / required evidence |
|---|---|---|---|---|---|---|---|
| Jiang, Yao, Tan, **OSR**, NeurIPS 2023 | Offline RL; learns inverse dynamics, not a forward model | Recovery means returning an out-of-sample state transition toward the dataset transition distribution | Policy route minimizes a KL from inverse-dynamics actions to the actor; value route contrasts Q-values of policy actions and inverse-dynamics target actions. The task Bellman residual remains standard; no full modified-operator analysis is given. | Theorem 1 equates action-distribution matching and successor-distribution matching under bounded actions plus continuity/positivity assumptions. Theorem 2 gives approximate value-level equivalence with an error controlled by a continuity gap. | Bayes' rule, KL identities, continuity bounds; distributional/approximate, not a safety certificate | Translating action constraints into successor-distribution statements; actor-versus-critic formulations; explicit approximation error | Offline behavior support, inverse dynamics, and dataset recovery do not imply collision avoidance or station return. Required evidence: successor-distribution calibration and perturbation sensitivity, not safety claims. |
| Tiofack et al., **Guided Flow Policy**, ICLR 2026 | Offline RL; model-free; critic, one-step actor, and flow BC policy trained jointly | Dataset support and high-value action preference, not physical safety | Standard Bellman MSE for critic; actor minimizes \(-Q+\alpha\|a_\theta-a_\omega\|^2\); flow model uses value-weighted flow matching. No new Bellman operator is derived. | No formally labeled theorem/proposition in the provided version. | Algorithmic loss design and ablations; empirical | Joint training, value-aware weighting, bidirectional guidance, and separating fast actor from richer proposal distribution | Value-aware imitation cannot certify constraints. Offline support assumptions and iterative flow sampling are unnecessary for the current online SAC setting. Required evidence: high-value candidate quality and stability ablations only. |
| Chae et al., **Flow Actor-Critic**, ICLR 2026 | Offline RL; flow behavior proxy gives samples and density | Confidence-weighted behavior support | Critic loss adds \(\alpha\mathbb E_{a\sim\pi}[w_{\hat\beta}Q]\) to Bellman MSE. Pointwise stationarity yields a Bellman operator with additive pessimistic shift \(-\frac\alpha2w\pi/\beta\), and \(-\infty\) outside behavior support. | Proposition 1 derives the operator; Proposition 2 gives \(\gamma\)-contraction for fixed support-constrained \(\pi\); Theorems 1–2 claim conservative fixed point and support-constrained policy under \(\operatorname{supp}\hat\beta\subseteq\operatorname{supp}\beta\). | First-order stationarity, contraction, resolvent/occupancy argument; deterministic tabular operator claims conditional on support assumptions | The methodology loss \(\to\) stationarity \(\to\) operator \(\to\) fixed point \(\to\) actor support | Behavior density has no safety meaning online. The published fixed-point simplification needs correction: a resolvent applied to a penalty vector is generally not a pointwise occupancy times the local penalty. High-confidence current actions are unbiased only if all future visited penalties vanish. Required evidence: operator residuals and support leakage. |
| Luo, Ma, **CRABS**, NeurIPS 2021 | Online model-based safe RL; deterministic dynamics enclosed by a calibrated set-valued learned model | Neural discrete-time barrier certificate; viable versus irrecoverable states | Min-max barrier loss maximizes successor violation over the barrier superlevel set and model envelope; policy is adversarially trained to remain certified. SAC modification is secondary and does not define the certificate. | Requirements R1–R3 imply forward invariance by induction if the model envelope is calibrated and global maximization succeeds. The paper openly states theoretical model calibration under domain shift is unresolved; ensembles and MALA are heuristics. | Set invariance, induction, adversarial optimization; deterministic conditional guarantee, empirical in implementation | Clear separation of training, calibrated successor set, adversarial counterexamples, and certified policy; directly supports corridor-wise CEGIS | Global adversarial search and ensemble calibration do not automatically verify rolling set updates or the full UAV zonotope predecessor. Required evidence: envelope containment, optimizer falsification rate, and continuous-region verification. |
| Chow et al., **A Lyapunov-based Approach to Safe RL**, NeurIPS 2018 | CMDP with transient/terminal MDP; model-based DP theory plus approximate RL | Expected cumulative constraint cost and Lyapunov-induced feasible policies | Starts from \(\min_\pi C^\pi(x_0)\) subject to \(D^\pi(x_0)\le d_0\). A one-step Lyapunov inequality defines a statewise policy set; minimizing Bellman backup over that set gives a safe Bellman operator. | Uniformly bounded hitting time makes policies proper and supports contraction. Safe policy iteration preserves feasibility, improves cost, and converges under tie-breaking regularizers. Optimality needs a strong baseline-to-optimum proximity assumption. | Lyapunov dominance, monotone contraction, fixed-point and policy-iteration arguments; expected constraint guarantee | Baseline certified recovery policy; one-step inequality turning a global budget into local feasible actions; policy-update feasibility | Expected CMDP constraints do not imply pathwise collision safety or worst-case return energy. Uniform bounded stopping for every stationary policy is too strong. Required evidence: stopping-time and Lyapunov residual bounds. |
| Chen et al., **Backup Control Barrier Functions**, CDC 2021 | Continuous-time known control-affine dynamics; fixed backup controller | States whose backup trajectory stays safe and reaches a known invariant backup set | Defines a finite-horizon constrained reachable set and an implicit barrier as the minimum margin along the backup flow and at the terminal backup set; a QP enforces all barrier conditions. | The reachable set is control invariant; the backup CBF QP is always feasible on it under backup-set invariance; finite time-grid approximation needs Lipschitz error bounds. | Flow semigroup, set invariance, sensitivity Jacobians, sufficient QP constraints; deterministic | Strongest structural analogue for the frozen UAV recovery policy and certified corridor; directly supplies backup-action existence | Continuous-time control-affine QP assumptions do not match discrete uncertain residual dynamics or latent memory. Required evidence: discrete successor enclosure and time/state discretization residuals. |
| Hsu, Nguyen, Fisac, **ISAACS**, L4DC 2023 | Model-based robust RL with known bounded disturbance set | HJ-Isaacs robust safety value and failure-avoidance set | HJI game uses \(V=\max_u\min_d\min\{g,V\circ f\}\). Adversarial SAC approximates a discounted counterpart. Strict safety comes later from a robust forward-reachable-set rollout, not from the learned critic. | Theorem 1 gives \(H+1\)-step safety when the robust rollout criterion holds. Infinite-horizon safety needs reachability of a robust controlled-invariant terminal set. | Dynamic games, reachability, zonotope/FRS propagation, receding-horizon induction; deterministic robust conditional guarantee | Learned field/policy as an untrusted oracle; bounded successor tube as certificate; fallback rollout and terminal invariant set | Requires full observability, known dynamics/Jacobians, and a valid disturbance set. Learned HJ values alone are explicitly uncertified. Required evidence: FRS containment, Taylor remainder bounds, boundary confusion plots. |
| Bertsekas, Tsitsiklis, **Stochastic Shortest Path theory**, MOR 1991; Bertsekas, **Proper Policies in Infinite-State SSPs**, 2017 | Undiscounted first-passage control; finite or infinite state | Proper policy reaches terminal in finite expected time; nonnegative stage cost in the infinite-state treatment | Bellman evaluation is \(J^\kappa=c^\kappa+P^\kappa J^\kappa\), with zero terminal value; control uses the corresponding minimum operator. | Existence of a proper policy and exclusion or infinite cost of improper policies yield standard uniqueness results in finite SSPs. In infinite spaces, Bellman equations may have multiple solutions; uniqueness requires a specified function class and additional boundedness/uniform-properness conditions. | Monotone convergence, perturbation by positive stage cost, Lyapunov-like function classes, semicontractive DP; probabilistic expected-cost results | Correct foundation for the energy field; explains why \(\gamma<1\) cannot be imported and why finite-level descent is valuable | Expected properness is weaker than worst-case finite-time recovery. The robust supremum recursion needs a separate max-plus/acyclic argument. Required evidence: hitting-time tails, level descent, and cumulative cost. |
| Bharadhwaj et al., **Conservative Safety Critics**, ICLR 2021 | Online/off-policy model-free RL | Discounted/episodic probability of catastrophic failure | Learns a failure critic with a reversed-CQL regularizer to overestimate failure; actor maximizes reward subject to an expected safety-critic constraint and KL trust region; rollout uses rejection sampling. | Theorem 1 bounds failure probability with high probability using critic overestimation, sampling error, and policy KL; later results give performance and sublinear cumulative-failure bounds. | Conservative value bounds, performance-difference/KL arguments, concentration; probabilistic, nonzero failure tolerance | Explicit safety-critic semantics, error propagation into final risk, rejection sampling in continuous actions | Cannot give robust zero collision without structural assumptions. If no sampled action passes, it executes the least unsafe action. Discounted failure can hide late failures. Required evidence: calibration, underestimation tails, rejection exhaustion. |
| Suttle et al., **Sampling-based Safe RL for Nonlinear Dynamical Systems**, AISTATS 2024 | Online policy-gradient RL; deterministic dynamics in main theory; exact state-dependent safe action set assumed known | Hard one-step invariant action set \(C(x)=\{u:T(x,u)\in S\}\) | Defines the truncated policy \(\pi^C_\theta(u|x)=\pi_\theta(u|x)/\pi_\theta(C(x)|x)\) on \(C(x)\), derives its score correction, and optimizes expected discounted return with random-horizon policy gradients. | Positive safe-set volume, positivity/differentiability, reachability, and ergodicity assumptions yield a well-defined objective and convergence to stationary points. Every sampled action is safe because support is truncated. | Measure theory, Radon–Nikodym arguments, policy-gradient theorem, stochastic approximation; deterministic hard safety conditional on exact \(C(x)\) | Most direct literature support for state-dependent support constraints and the normalization correction in continuous action spaces | Assumes exact known membership and positive volume; uses on-policy policy gradients rather than off-policy SAC. Required evidence: feasible-set volume, sampler acceptance, and membership soundness. |
| Stolz et al., **Excluding the Irrelevant: Continuous Action Masking**, NeurIPS 2024 | Online policy-gradient RL with continuous actions; relevant sets are supplied from task knowledge and represented by convex sets, especially zonotopes | A state-dependent relevant-action set \(A_r(s)\subseteq A\); it is safe only when an independent verification procedure has made \(A_r\) a verified safe-action set | Defines a relevant policy as a pushforward/truncation of the nominal policy. Ray and generator masks map latent actions into \(A_r\); the distributional mask uses \(\pi^r_\theta(a|s)\propto\mathbf1_{A_r(s)}(a)\pi_\theta(a|s)\). No Bellman operator is modified. | Assumes convex \(A_r\) with computable center/boundary; the generator mask additionally assumes a fixed-generator zonotope and Gaussian policy. Propositions 1--5 derive PPO policy gradients or transformed distributions. Exact membership follows from the mapping, conditional on the supplied set and exact computation. | Change of variables, affine Gaussian transformation, score-function differentiation, bijectivity; exact conditional membership, not independent safety certification | **Complements Route A:** supplies concrete continuous mechanisms whose outputs lie in a given convex set and exposes the normalization/Jacobian terms required for mathematically correct learning | Does not construct or verify the safe set, does not analyze SAC, cannot directly handle disjoint/nonconvex sets, and approximates the intractable distributional-mask normalization gradient in experiments. Thus it supports membership enforcement, not the UAV certificate. Required evidence: set-inclusion verification, numerical membership residual, latency, and SAC-specific gradient validity. |
| Yu, Xu, Zhang, **SEditor**, NeurIPS 2022 | Online off-policy model-free CMDP; utility maximizer and learned safety-editor policies are trained from scratch without an oracle safety model | Discounted expected constraint reward / average violation-rate target, not a robust invariant set | Standard utility and constraint Bellman backups. A primal-dual surrogate is split so the utility actor maximizes task \(Q\), while the editor maximizes constraint \(Q_c\) minus a hinge utility-loss term; the final action is an additive learned edit. No modified safety Bellman operator or formal certificate theorem is supplied. | The paper explicitly accepts unavoidable violations during learning and asymptotic budget satisfaction. It reports violation rates as low as \(5\times10^{-4}\), but contains no labeled theorem, proposition, lemma, or proof establishing zero violation or action-set inclusion. | First-order SGD, reparameterization, empirical ablation; statistical performance only | **Complements empirical design:** useful as a proposal/editor baseline and for studying whether a separate learned editor improves utility--violation tradeoffs | **Cannot enter the strict proof chain:** low expected violation does not imply \(a_{\rm exec}\in\mathcal A_{\rm cert}(z)\), recursive feasibility, or safe return. It may be composed before a certified mask/projection, but then the latter—not SEditor—provides the guarantee. Required evidence: violation confidence intervals and editor ablations, reported only as empirical results. |
| Markgraf et al., **Safe RL using Action Projection**, TMLR 2026 | Actor-critic RL with a closest-point projection safeguard, analyzed as safe-environment RL (SE-RL) or safe-policy RL (SP-RL) | A pre-characterized nonempty state-dependent safe set \(U_x^\phi\); projection solves a convex closest-point problem into this set | SE-RL composes projection with environment transitions/rewards; SP-RL makes the projected pushforward policy explicit. Theorem 1 shows equality of optimal state values; Lemma 1 and Corollary 1 give update equivalence only for stochastic-policy algorithms using GAE. Lemmas 2--3 characterize action aliasing and a flat critic. | Requires a safe set whose actions admit continued constraint satisfaction, nonemptiness on \(\widetilde X\), convexity for unique projection, and exact safeguard solution. SAC is explicitly outside the formal analysis. | Pushforward MDP comparison, law of total probability, normal-cone geometry, implicit differentiation; hard action membership conditional on the supplied set | **Complements runtime enforcement and modifies assumptions:** projection can make executed actions belong to a verified set, but replay/critic semantics must reflect whether projection is inside the policy or environment | Projection does not verify the set. Many-to-one aliasing can flatten an SE-RL critic and create rank-deficient SP-RL Jacobians; stochastic projected densities may contain boundary singularities. Its value-equivalence theorem cannot be imported as a SAC policy-gradient theorem. Required evidence: projection feasibility/tolerance, fallback frequency, aliasing diagnostics, and executed-action logging. |

### Cross-literature map

The ten core sources do not cover the entire proof stack. The following adjacent primary sources fill the remaining roles:

| Direction | Representative source and usable lesson |
|---|---|
| CBF and discrete-time CBF | [Ames et al., CBF-QP](https://doi.org/10.1109/TAC.2016.2638961) gives continuous-time invariance; [Cosner et al., robust stochastic DTCBF](https://arxiv.org/abs/2302.07469) shows that stochastic discrete-time conditions produce finite-horizon probability bounds, not automatic robust invariance. |
| Neural barriers and formal verification | [Exact Verification of ReLU Neural CBFs](https://proceedings.neurips.cc/paper_files/paper/2023/file/120ed726cf129dbeb8375b6f8a0686f8-Paper-Conference.pdf) uses nonsmooth invariance conditions, partitioning, IBP, and linear relaxation. |
| Verification-in-the-loop / CEGIS | [Wang et al., verification-in-the-loop neural CBF synthesis](https://arxiv.org/abs/2311.10438) returns uncertified partitions to training; this matches the planned counterexample buffer. |
| CPO and CVPO | [CPO](https://proceedings.mlr.press/v70/achiam17a.html) and [CVPO](https://proceedings.mlr.press/v162/liu22b.html) constrain expected cumulative costs and policy updates. They are optimization baselines, not pathwise collision certificates. |
| Lyapunov SAC | [Lyapunov-based Safe Policy Optimization for Continuous Control](https://arxiv.org/abs/1901.10031) adapts Lyapunov feasibility to continuous policies, but its guarantee remains tied to the chosen CMDP constraint semantics and approximation quality. |
| HJ reachability and residual certification | ISAACS supplies learned HJ plus reachable tubes; [Certifying HJ Reachability Learned via RL](https://arxiv.org/abs/2602.16475) is a current preprint using residual verification/SMT ideas. It is relevant to certification tooling but is not yet a peer-reviewed foundation. |
| Viability kernels | [Viability theory survey](https://epubs.siam.org/doi/10.1137/0328044) identifies the viability kernel as the largest controlled-invariant subset; this is the right global object for inevitable-collision semantics, but exact computation is generally intractable. |
| MPC recursive feasibility | Backup CBF and robust MPC share the shifted-feasible-plan argument: a terminal invariant set plus a feasible backup sequence ensures a successor feasible sequence. This is the proof pattern needed for the recovery corridor. |
| Budget-state augmentation | [Sauté RL](https://proceedings.mlr.press/v162/sootla22a.html) converts an almost-sure budget constraint into an augmented-state problem. This validates keeping remaining energy outside the recovery-cost field while including it in the decision state. |
| Reach-avoid RL | ISAACS and HJ-RL methods use non-additive min/max Bellman/Isaacs recursions. They should not be replaced by an ordinary additive reward without proving semantic equivalence. |
| Modified Bellman operators from regularization | [CQL](https://proceedings.neurips.cc/paper/2020/file/0d2b2061826a5df3221116a5085a6052-Paper.pdf) and FAC show how a critic loss induces a shifted operator. The penalty's resolvent propagation, not its local value alone, determines bias. |
| Two-timescale actor-critic | [Konda and Tsitsiklis](https://doi.org/10.1137/S0363012901385691) and [COF-PAC](https://proceedings.mlr.press/v119/zhang20s.html) prove convergence under restrictive linear-critic/stochastic-approximation assumptions. They do not directly justify joint deep SAC, model, field, generator-construction, and certificate updates. |

## Derivation Strategy

The comparison starts from an explicit optimization or loss for each route:

1. Derive the state-dependent constrained soft operator as the ideal Route-A problem, then derive the exact affine-tanh generator density and actor gradient for the selected implementation.
2. Derive two field-weighted critic operators from pointwise first-order stationarity: a linear pessimistic loss and a quadratic floor-anchoring loss.
3. Define distinct task, failure/reachability, and recovery-energy critics; analyze each according to discounted, absorbing, or SSP semantics.
4. Separate RL fixed-point results from the invariant-set and finite-time recovery proof.
5. Compare the routes only under frozen certified objects; then state why changing fields breaks the fixed-operator proof.

## Derivation Map

1. Sound free/obstacle/unknown and return-corridor updates plus the real uncertainty envelope \(\Rightarrow\) a sound explicit certificate-state successor set.
2. Frozen-\(\kappa\) corridor transition plus one-step or bounded-\(M\)-step progress \(\Rightarrow\) finite-time terminal reachability.
3. Progress plus the robust energy recursion \(\Rightarrow\) unique finite transit-energy value and a verified upper bound.
4. Collision, corridor, energy, and progress verification \(\Rightarrow\) corridor-wide certification of \(\kappa\).
5. Certified \(\kappa\) plus complete-zonotope predecessor verification \(\Rightarrow\) \(c(z)+G(z)[-1,1]^3\subseteq\mathcal A_{\rm cert}(z)\).
6. The affine-tanh generator plus rank/inclusion/deadline checks and \(\kappa\) fallback \(\Rightarrow\) every executed action belongs to \(\mathcal A_{\rm cert}(z)\).
7. Executed-action membership \(\Rightarrow\) invariance, recursive feasibility, and switching safety by induction.
8. In parallel, a frozen measurable correspondence plus an explicit RL objective \(\Rightarrow\) route-specific actor and Bellman results.

## Main Derivation

### Route A: state-dependent support-constrained SAC

#### 1. Why the problem exists

A squashed Gaussian SAC policy has positive density throughout the interior of its actuator box. Even if dangerous actions have very low probability, strict safety requires zero probability outside the certified set. Penalizing expected violations cannot provide this support property.

#### 2. Literature treatment

Sampling-based safe RL truncates a policy to a known state-dependent action set and derives the normalization correction. Stolz et al. provide three continuous-set realizations: a bijective ray map, a zonotope generator map, and a distributional truncation. Markgraf et al. distinguish projection inside the environment from projection inside the policy and show that their optimal state values may coincide while their learning dynamics can differ because of action aliasing. Regularized-MDP and SAC theory derive Boltzmann improvement over a fixed action space. The missing step is still UAV-specific: neither masking nor projection constructs the verified action correspondence, and neither cited paper derives the required off-policy soft actor update for a state-dependent projected SAC policy.

#### 3. Symbols and measurable domain

Let \(\mathcal Z_G\subseteq\mathcal S_{\rm joint}^{\rm cert}\) denote the generator-enabled certificate states on which the pure Route-A operator is analyzed. The selected runtime construction is

\[
C_{\rm run}(z)=c(z)+G(z)[-1,1]^3,
\qquad
C(z):=\operatorname{int}C_{\rm run}(z),
\tag{RA1}
\]

with \(G(z)\in\mathbb R^{3\times3}\), \(\sigma_{\min}(G(z))\ge\sigma_G>0\), and \(C_{\rm run}(z)\subseteq\mathcal A_{\rm cert}(z)\cap\mathcal A\). Freeze the feasible correspondence

\[
C:\mathcal Z_G\rightrightarrows\mathcal A,
\qquad
\varnothing\neq C(z)\subseteq\mathcal A_{\rm cert}(z),
\qquad
\mathsf G_C:=\operatorname{Gr}(C)
=\{(z,a):z\in\mathcal Z_G,\;a\in C(z)\}.
\]

The graph \(\mathsf G_C\) is measurable because \(c\) and \(G\) are measurable. Let \(\lambda_A\) be three-dimensional Lebesgue measure, \(\alpha>0\), and \(Q\in\mathbb B_b(\mathsf G_C)\). Full rank gives

\[
\lambda_A(C(z))
=8|\det G(z)|
\ge8\sigma_G^3>0.
\tag{RA2}
\]

The compact actuator bound supplies a uniform upper bound. Thus the positive-measure requirement is a verified consequence of the selected construction, not an independent sampling assumption.

For each \(Q\) under consideration, assume

\[
0<Z_C^Q(z)
:=\int_{C(z)}\exp(Q(z,a)/\alpha)d\lambda_A(a)<\infty
\quad\forall z\in\mathcal Z_G.
\]

Pointwise positivity and finiteness make each statewise variational problem meaningful. To make \(\mathcal T_C\) a self-map on bounded functions, additionally require \(0<m_C\le\lambda_A(C(z))\le M_C<\infty\) uniformly over certified states. This condition is stronger than nonemptiness supplied by the single fallback action.

Because \(Q\) is trace-measurable on the measurable graph, its zero extension is measurable on \(\mathcal Z_G\times\mathcal A\). Hence \(\mathbf1_{\mathsf G_C}(z,a)e^{Q(z,a)/\alpha}\) is jointly measurable. Measurability of parameterized integrals implies that \(z\mapsto Z_C^Q(z)\) and \(z\mapsto V_C^Q(z)=\alpha\log Z_C^Q(z)\) are measurable, while

\[
\pi_C^Q(B|z)
=\frac{\int_{B\cap C(z)}e^{Q(z,a)/\alpha}d\lambda_A(a)}
{Z_C^Q(z)}
\]

is a measurable stochastic kernel.

Separately, let \(u\in\mathbb R^3\) be the Gaussian pre-squash actor output, let \(\eta=\tanh u\), and let \(a_M=c(z)+G(z)\eta\). With fail-safe kernel \(M_z^{\rm fs}\), the complete executed policy is

\[
\pi_{\rm exec}(B|z)
=\int_{\mathbb R^3}
M_z^{\rm fs}(B|u)\,
\mu_\theta(du|o^{\rm task})
\quad\text{for measurable }B\subseteq\mathcal A.
\]

The variational derivation below concerns the full feasible density class on \(C\). T12A proves the exact density and gradient of the selected generator realization. T8A--T9A apply to that realization only under their stated policy-class realizability condition. If fallback executes \(\kappa(z)\notin C(z)\), T14A's mixed reference measure—not the pure feasible-density objective—describes the complete law.

#### 4. Starting optimization

The pointwise policy-improvement problem is

\[
\begin{aligned}
\max_{\pi(\cdot|z)}\;&
\int_{C(z)}\pi(a|z)Q(z,a)\,d\lambda_A(a)
-\alpha\int_{C(z)}\pi(a|z)\log\pi(a|z)\,d\lambda_A(a)\\
\text{s.t. }&\pi(a|z)\ge0,\qquad
\int_{C(z)}\pi(a|z)\,d\lambda_A(a)=1,\\
&\pi(a|z)=0\quad\lambda_A\text{-a.e. on }\mathcal A\setminus C(z).
\end{aligned}
\]

#### 5. Step-by-step derivation

Introduce a multiplier \(\eta(z)\) for normalization. On \(C(z)\), the variational derivative is

\[
Q(z,a)-\alpha(\log\pi(a|z)+1)+\eta(z)=0.
\]

Thus

\[
\pi(a|z)=\exp\!\left(\frac{Q(z,a)}{\alpha}\right)
\exp\!\left(\frac{\eta(z)-\alpha}{\alpha}\right).
\]

Normalization gives the partition function

\[
Z_C^Q(z)=\int_{C(z)}\exp\!\left(\frac{Q(z,b)}{\alpha}\right)d\lambda_A(b),
\]

and therefore the unique optimizer, up to null sets,

\[
\boxed{
\pi_C^Q(a|z)=
\frac{\mathbf 1_{C(z)}(a)\exp(Q(z,a)/\alpha)}{Z_C^Q(z)}.}
\]

This density is a distributional mask of the unconstrained Boltzmann density. A distributional mask applied to an arbitrary Gaussian has the same support property but is not generally the optimizer above. Likewise, a ray, generator, or closest-point map produces a pushforward distribution on \(C(z)\), not necessarily this truncated Boltzmann distribution.

The optimized pointwise value is the convex conjugate of negative entropy:

\[
V_C^Q(z)=\alpha\log Z_C^Q(z).
\]

#### 6. Operator and fixed point

Let \(r:\mathsf G_C\to\mathbb R\) be bounded measurable and let \(P(\cdot|z,a)\) be a measurable stochastic kernel supported on \(\mathcal Z_G\). This support condition belongs only to the pure Route-A operator; transitions leaving \(\mathcal Z_G\) invoke the hybrid fallback process. For frozen \(C\), define

\[
(\mathcal T_C Q)(z,a)
=r(z,a)+\gamma\,\mathbb E_{z'\sim P(\cdot|z,a)}
\left[\alpha\log\int_{C(z')}e^{Q(z',b)/\alpha}d\lambda_A(b)\right].
\]

This equation is defined only for \((z,a)\in\mathsf G_C\); values of infeasible actions are neither required nor assigned an optimality meaning. Under A10a--A10d, \(\mathcal T_C\) maps \(\mathbb B_b(\mathsf G_C)\) into itself.

For bounded \(Q_1,Q_2\), log-sum-exp over the same nonempty set is 1-Lipschitz in sup norm:

\[
|V_C^{Q_1}(z)-V_C^{Q_2}(z)|\le\|Q_1-Q_2\|_\infty.
\]

Hence

\[
\|\mathcal T_CQ_1-\mathcal T_CQ_2\|_\infty
\le\gamma\|Q_1-Q_2\|_\infty.
\]

Banach's fixed-point theorem gives a unique constrained soft-optimal \(Q_C^*\in\mathbb B_b(\mathsf G_C)\). A constrained soft policy-improvement theorem follows only when the old and new policies use the same frozen \(C\), the same \(P\) and \(r\), the same \(\lambda_A\), and the same feasible density class. The pointwise variational maximizer is then followed by monotone evaluation in that unchanged control problem.

#### 6a. Selected runtime realization and retained controls

The support theorem and the soft-improvement theorem must not be conflated:

1. **Primary implementation: full-rank three-generator map.** T12A derives \(u\mapsto c+G\tanh u\), including the Gaussian, tanh, and affine Jacobians. The image is \(C(z)\), while its closure is \(C_{\rm run}(z)\).
2. **Ideal baseline: exact truncation.** T11A remains the unrestricted feasible-density baseline and experimental control. It is not the main implementation because its normalizer and constrained sampler are unnecessary for the affine generator.
3. **Projection control.** T13A remains an environment-side or policy-side control for studying aliasing and induced reference measures. It is not part of the main theorem path.
4. **Extensions only.** Generator counts \(g>3\), noninjective matrices, rank loss, and dimension-changing maps are not covered by T12A's determinant formula. They revert to T10A and require induced-measure or coarea analysis.
5. **SEditor control.** A learned edit \(E_{\rm edit}(z,u,\Delta u)\) has no support guarantee unless its complete image is independently verified inside \(C_{\rm run}(z)\). It remains an empirical proposal baseline.

#### 7. What this proves

For a frozen, exact \(C(z)\), it gives a genuinely different Bellman operator, a normalized maximum-entropy policy supported only on \(C(z)\), contraction, a unique fixed point, and policy improvement within the fixed feasible class. Separately, a sound mask or projection proves only that its executed pushforward policy has support in \(C(z)\); it inherits the soft fixed-point and improvement conclusions only when its actual policy class and density solve the stated variational problem.

#### 8. What this does not prove

- It does not prove that the learned fields define a sound \(C(z)\).
- It does not prove that a sampler exactly realizes the truncated density.
- It does not prove that a mask or projection is numerically total, meets real-time deadlines, or stays feasible under solver tolerance.
- It does not prove that the nominal-policy entropy equals the executed-policy entropy after a many-to-one map.
- It does not prove soft optimality of the full fail-safe law when fallback adds a Dirac mass at \(\kappa(z)\notin C(z)\).
- Markgraf et al.'s SE-RL/SP-RL optimal-value equivalence is not a SAC gradient-equivalence theorem; their analysis explicitly excludes SAC modifications.
- SEditor does not prove support inclusion, zero violations, recursive feasibility, or return-to-station capability.
- If \(C(z)\) has zero Lebesgue measure, the density above is undefined; a lower-dimensional or discrete reference measure is required.
- If \(C_k\) changes during training, \(\mathcal T_{C_k}\) and \(\mathcal T_{C_{k+1}}\) are different operators. If the set shrinks, the old policy may be infeasible and ordinary policy improvement is inapplicable. If it expands, the old policy remains feasible, but value comparison still requires evaluation under the same frozen transition/reward model.
- If the chosen mask/projection induces only a restricted pushforward-policy class, exact optimality over all densities supported on \(C(z)\) is not established.

#### 9. Connection to the UAV method

The certified action correspondence contains accelerations whose complete certificate-state successor envelope remains in verified local free geometry and the sparse return corridor while preserving collision and energy margins. A continuous verifier constructs \(c,G\) so that the complete zonotope is an inner approximation. The feedforward actor chooses only \(\eta\); it cannot enlarge or certify the set. The frozen recovery action supplies fallback nonemptiness independently of the zonotope.

#### 10. Required implementation and evidence

Membership must be sound over the complete zonotope and continuous successor sets. Verification must report \(\sigma_{\min}(G)\), \(|\det G|\), zonotope volume, set-inclusion residuals, construction time, rejection/fallback rates, and runtime latency. Replay stores \(u,\eta,c,G,a_M,a_{\rm exec}\), and acceptance. T12A's exact log density is used for the accepted generator branch; T14A is used when fallback events are part of the optimized law. All RL theorems freeze \(c,G\), the corridor, and verified envelopes within each policy-improvement argument.

### Route B: field-weighted conservative critic

#### 1. Why the problem exists

This route attempts to discourage uncertified actions without solving a hard constrained policy problem. Its theoretical question is not whether a weight can be added, but which Bellman operator is implied by the chosen critic loss and whether that operator can force actor support.

#### 2. Literature treatment

FAC and CQL derive conservative values from regularized critic objectives. FAC is especially relevant because it differentiates a pointwise tabular loss. Its proof pattern transfers, but behavior density must be replaced by a certification quantity with a clear measure-theoretic role.

#### 3. Symbols

Let \(d(z)>0\) be the replay state density at the point being differentiated, \(\mu(a|z)>0\) the replay action density, \(\nu(a|z)\) a proposal density (usually the actor), \(w(z,a)\ge0\) a frozen uncertification weight, and \(\lambda\ge0\) the fixed regularization coefficient. The quadratic alternative also uses a finite floor \(q_{\min}\in\mathbb R\). Let

\[
y_Q(z,a)=r(z,a)+\gamma\mathbb E_{z',b\sim\pi}[Q(z',b)-\alpha\log\pi(b|z')]
\]

be a frozen-target soft Bellman target during pointwise differentiation.

#### 4. Starting optimization: linear pessimism

Consider

\[
\mathcal L_{\rm lin}(Q)
=\frac12\mathbb E_{d(z)\mu(a|z)}(Q-y_Q)^2
+\lambda\mathbb E_{d(z)\nu(a|z)}[wQ].
\]

#### 5. Stationary derivation

At a tabular pair with \(d(z)>0\),

\[
\frac{\partial\mathcal L_{\rm lin}}{\partial Q(z,a)}
=d(z)\mu(a|z)(Q-y_Q)+\lambda d(z)\nu(a|z)w(z,a).
\]

Setting this to zero yields

\[
Q=y_Q-\lambda\frac{\nu(a|z)}{\mu(a|z)}w(z,a).
\]

Thus, for fixed \(\pi,w,\mu,\nu\),

\[
\boxed{\mathcal T_{\rm lin}^{\pi,w}Q
=\mathcal B_\alpha^\pi Q-\lambda c_w,\qquad
c_w=\frac{\nu}{\mu}w.}
\]

Let \(P^\pi\) be the policy-induced state-action transition operator and \(Q_\alpha^\pi\) the unique fixed point of the ordinary soft evaluation operator \(\mathcal B_\alpha^\pi\). The shifted operator is monotone and a \(\gamma\)-contraction because its additive penalty is independent of \(Q\). Its unique fixed point is

\[
\boxed{
\bar Q^{\pi,w}=Q_\alpha^\pi-\lambda(I-\gamma P^\pi)^{-1}c_w.
\]

Equivalently, the bias at a starting pair is the expected discounted sum of future penalties:

\[
Q_\alpha^\pi(z,a)-\bar Q^{\pi,w}(z,a)
=\lambda\mathbb E_\pi\left[\sum_{t\ge0}\gamma^t c_w(z_t,a_t)\mid z_0=z,a_0=a\right].
\]

This correct resolvent form exposes an important FAC caveat: zero local penalty at \((z,a)\) does **not** imply an unbiased value if future policy occupancy reaches penalized pairs.

If \(\mu=0\) but \(\nu w>0\), the linear objective is unbounded below as \(Q\to-\infty\); a finite neural network only approximates this behavior and does not create exact \(-\infty\) support exclusion.

#### 6. Alternative operator from a quadratic floor anchor

To avoid an unbounded linear objective, consider

\[
\mathcal L_{\rm quad}(Q)
=\frac12\mathbb E_{d\mu}(Q-y_Q)^2
+\frac\lambda2\mathbb E_{d\nu}[w(Q-q_{\min})^2].
\]

Pointwise stationarity gives

\[
Q=\frac{\mu y_Q+\lambda\nu wq_{\min}}{\mu+\lambda\nu w}.
\]

Define \(\beta_w=\mu/(\mu+\lambda\nu w)\in[0,1]\). The induced operator is

\[
\boxed{
\mathcal T_{\rm quad}^{\pi,w}Q
=\beta_w\mathcal B_\alpha^\pi Q+(1-\beta_w)q_{\min}.}
\]

It is monotone, and

\[
\|\mathcal T_{\rm quad}Q_1-\mathcal T_{\rm quad}Q_2\|_\infty
\le\gamma\|\beta_w\|_\infty\|Q_1-Q_2\|_\infty.
\]

Thus it has a unique fixed point for frozen weights. This is a genuine field-weighted Bellman operator derived from a loss, but it changes task values even in regions whose future trajectories later approach uncertified actions.

#### 7. What this proves

Both losses yield explicit monotone contractions and unique fixed points under frozen densities and weights. They characterize exactly how field penalties propagate backward through future occupancy.

#### 8. What this does not prove

The full-action maximum-entropy actor for any finite \(\bar Q\) is proportional to \(e^{\bar Q/\alpha}\) and therefore has positive density wherever \(\bar Q\) is finite. Consequently, finite critic conservatism does not imply strict support exclusion, recursive feasibility, collision invariance, or finite-time recovery. Making \(Q=-\infty\) outside the certified set collapses this route into Route A's hard support constraint.

#### 9. Connection to the UAV method

Field-weighted critics can be useful auxiliary shaping: they can reduce proposals near low-confidence boundaries and improve sampling efficiency. They cannot be the sole carrier of a robust zero-violation theorem.

#### 10. Required implementation and evidence

The replay/proposal density ratio must be controlled; weights must be frozen during evaluation; operator residual and bias propagation should be tested; unsafe-action probability must be measured directly; and hard execution must still use a sound certified action membership test if strict safety is claimed.

### Route C: separate task and safety/recovery critics

#### 1. Why the problem exists

Collision failure probability, reachability margin, and return energy are mathematically different quantities. Combining them into one scalar task critic hides their boundary conditions, discount semantics, and error propagation.

#### 2. Literature treatment

CSC learns a discounted/episodic failure critic; HJ-RL learns a non-additive reachability value; Lyapunov methods learn cumulative constraint values; SSP theory treats undiscounted first-passage cost. These operators are not interchangeable.

#### 3. Symbols and candidate critics

The task critic uses the usual soft evaluation operator

\[
Q_R^\pi(z,a)=r(z,a)+\gamma_R\mathbb E[V_R^\pi(z')].
\]

A discounted failure critic is

\[
Q_F^\pi(z,a)
=\mathbb E_\pi\left[\sum_{t\ge0}\gamma_F^t\mathbf1_{\mathcal F}(z_{t+1})\mid z_0=z,a_0=a\right].
\]

If failure is absorbing and counted once, \(\gamma_F=1\) gives failure probability only when termination/properness makes the equation well posed. For \(\gamma_F<1\), it is a time-discounted failure score, not the probability of eventual failure.

A robust reachability critic uses a non-additive recursion such as

\[
V_{\rm reach}(z)=\max_a\min_{z'\in\operatorname{Post}(z,a)}
\min\{g(z),V_{\rm reach}(z')\}.
\]

The recovery-energy critic for frozen \(\kappa\) is

\[
R^\kappa(z^{\rm cert})=
\begin{cases}
0,&z\in\mathcal G,\\
\bar c(z,\kappa(z))+\sup_{z'\in\operatorname{Post}(z,\kappa(z))}R^\kappa(z'),&z\notin\mathcal G.
\end{cases}
\]

#### 4. Starting actor optimization

A generic dual-critic actor problem is

\[
\max_\pi\;\mathbb E[Q_R^\pi-\alpha\log\pi]
\quad\text{s.t.}\quad
Q_F(z,a)\le\delta_F,\quad
e-\bar c(z,a)\ge
\overline R^\kappa(\bar z^{{\rm cert}\prime})+e_G+m_{\rm res}(z')
\quad\text{for all admissible }z'.
\]

If constraints are enforced only in expectation under \(\pi\), the result is a CMDP-style average guarantee. If they define a pointwise feasible action set, the actor improvement again becomes Route A.

#### 5. Operator and uniqueness analysis

- For \(\gamma_F<1\), the failure Bellman operator is a contraction, but the semantics discount late failure.
- For \(\gamma_F=1\), uniqueness requires absorbing/proper-policy conditions or selection of the minimal nonnegative fixed point.
- HJ/reachability operators use min/max structure and require their own monotonicity and fixed-point arguments; an additive reward critic is not equivalent.
- The robust one-step recovery-energy operator is monotone but not generally a sup-norm contraction. Under A8-1, T4a uses direct level induction. Under A8-M, T4b uses the stopped block operator (E2); it does not reuse one-step induction or claim one-step contraction.

#### 6. Policy improvement

A candidate recovery policy may minimize the robust one-step expression, but replacement is safe only after verifying collision preservation, energy recursion, and level progress on the full certified region. A smaller empirical energy estimate is not sufficient. Task-policy improvement preserves feasibility only when the safety-critic upper/lower errors are known and incorporated into pointwise constraints.

#### 7. What this proves

This route gives each physical property the correct operator and makes approximation-error propagation explicit. It is the natural home for the undiscounted energy field and for probabilistic failure analyses.

#### 8. What this does not prove

A learned failure probability below a threshold does not imply robust zero collision. A Lagrangian update does not preserve pointwise feasibility. A robust energy critic alone does not define the task actor's support. Hard safety still requires a verified action correspondence or runtime shield.

#### 9. Connection to the UAV method

Even if Route A is selected as the task-RL backbone, the energy field should still be analyzed as the Route-C SSP critic of the frozen recovery policy. Thus “primary Route A” does not mean collapsing all values into one critic.

#### 10. Required implementation and evidence

Each critic needs separate labels, calibration, Bellman residuals, and boundary tests. Reports must distinguish discounted failure, eventual failure probability, expected energy, tail energy, and worst-case energy.

## Route Comparison and Recommendation

| Criterion | A. Support-constrained SAC | B. Field-weighted critic | C. Separate critics |
|---|---|---|---|
| Hard-safety provability | Highest, if the action set is sound and nonempty | Insufficient by itself; finite pessimism does not imply zero support | Medium; becomes high only when critic bounds define pointwise verified actions |
| Compatibility with current fields | Direct: fields define successor feasibility | Direct: margins become weights | Direct: collision/energy retain separate semantics |
| New Bellman operator | Yes: constrained soft log-partition operator | Yes: shifted or shrinkage operator derived from the actual loss | Task operator may remain standard; safety/recovery operators are genuinely distinct |
| Continuous 3-D action difficulty | The selected full-rank generator avoids online projection and truncation normalization, but requires real-time complete-zonotope verification | Lowest | Medium to high depending on constraint enforcement |
| Natural compatibility with SAC | High: T12A gives the exact affine-tanh log density and actor gradient; exact Boltzmann realizability remains conditional | High operationally | High for Lagrangian SAC, but hard safety needs pointwise constraints |
| Model-error bounds for strict safety | Required | Still required; critic penalty cannot replace them | Required for robust critics/action tests; statistical bounds suffice only for probabilistic claims |
| Recursive feasibility | Natural with frozen recovery action and successor corridor | Not implied | Possible if the safety/recovery critic is a verified predecessor or SSP upper function |
| Finite-time recovery | Added through recovery levels/SSP | Not implied | Natural through properness and level descent |
| Main theoretical risk | Time-varying feasible sets and continuous-action sampling | Mistaking pessimism for certification | Mixing incompatible safety-critic semantics or relying on critic calibration without verification |

**Fixed primary route:** Route A as the task-RL backbone, combined with a separate undiscounted SSP energy field for the frozen recovery policy. This produces the cleanest theorem separation: the certified action correspondence supplies hard feasibility; constrained soft RL supplies task optimality within that domain; the recovery level supplies finite-time arrival.

**Fixed fallback route:** Route C, with separate task, reachability/failure, and SSP energy critics. It preserves semantic clarity, but any strict theorem must convert critic bounds into a pointwise verified action set. A purely expectation-constrained dual-critic SAC is probabilistic/average constrained RL, not robust safety.

**Role of Route B:** auxiliary training heuristic or ablation, not the primary safety theorem. It is valuable for proposal efficiency and for a FAC-style operator contribution, but not for recursive feasibility.

## Runtime Safeguards

### Certificate source versus enforcement mechanism

The runtime chain has four logically different obligations:

\[
\underbrace{\text{verified dynamics/field envelopes}}_{\text{physical premises}}
\Longrightarrow
\underbrace{C_{\rm run}(z)\subseteq\mathcal A_{\rm cert}(z)}_{\text{certificate}}
\Longrightarrow
\underbrace{a_{\rm exec}\in\mathcal A_{\rm cert}(z)}_{\text{verified candidate or certified fallback}}
\Longrightarrow
\underbrace{z^+\in\mathcal S_{\rm joint}^{\rm cert}}_{\text{invariance conclusion}}.
\]

- The first two terms determine whether an action is genuinely certified.
- On successful candidate execution, the full-rank generator map plus the set-level verifier establishes \(a_M\in C_{\rm run}(z)\subseteq\mathcal A_{\rm cert}(z)\).
- On failure, timeout, infeasibility, or rejection, the separate recovery certificate establishes \(\kappa(z)\in\mathcal A_{\rm cert}(z)\); membership of \(\kappa(z)\) in \(C_{\rm run}(z)\) is not required.
- SEditor supplies neither implication. Without an additional verified range-inclusion result, it is an empirical proposal transformation whose reported low violation rate is not a certificate.

For the selected generator, let

\[
T_z(u)=c(z)+G(z)\tanh u.
\]

Conditional on a valid set-level certificate, the candidate kernel is \(M_z^{\rm cand}(\cdot|u)=\delta_{T_z(u)}\), and its mechanism-validity condition is

\[
M_z^{\rm cand}(C_{\rm run}(z)|u)=1.
\]

After including verification and fallback, define the executed-policy kernel by

\[
\pi_{\rm exec}(B|z)
=\int_{\mathbb R^3}
M_z^{\rm fs}(B|u)\mu_\theta(du|o^{\rm task}),
\]

where \(M_z^{\rm fs}\) returns the verified candidate on acceptance and \(\kappa(z)\) otherwise. The overall condition is \(M_z^{\rm fs}(\mathcal A_{\rm cert}(z)|u)=1\) for every certified state and pre-squash action \(u\in\mathbb R^3\). Candidate membership in \(C_{\rm run}\) and overall membership in \(\mathcal A_{\rm cert}\) are distinct conclusions.

### Selected generator and retained controls

| Role | Mechanism | Mathematical status | Guarantee boundary |
|---|---|---|---|
| Primary | \(a=c(z)+G(z)\tanh u\), \(G\in\mathbb R^{3\times3}\), \(\sigma_{\min}(G)\ge\sigma_G\) | T12A gives a bijection onto \(\operatorname{int}C_{\rm run}\), exact density, entropy, and SAC gradient | The map supplies membership only after continuous verification of the complete zonotope inclusion |
| Ideal baseline | Exact truncation on \(C_{\rm run}\) | T11A gives the unrestricted feasible-density optimum and normalizer gradient | Experimental/theoretical control; not the deployed actor |
| Projection control | Exact closest-point projection | T13A gives the safeguarded-MDP and induced-measure formulations | Projection supplies membership, not set validity |
| Empirical control | SEditor before the generator/verifier | No universal range inclusion follows from its loss | Only the following verified generator/fallback layer can provide strict safety |

### Fail-safe execution rule

The proof-compatible runtime rule is

\[
a_{\rm exec}=
\begin{cases}
a_M=c(z)+G(z)\tanh u,
&\begin{array}{l}
\text{if }z\in\mathcal Z_G\text{ and construction returns before the certified deadline,}\\
\sigma_{\min}(G)\ge\sigma_G,\;
C_{\rm run}(z)\subseteq\mathcal A_{\rm cert}(z),\\
\text{and all numerical and candidate checks pass},
\end{array}\\
\kappa(z), & \text{otherwise}.
\end{cases}
\tag{RS1}
\]

The first branch gives \(a_{\rm exec}\in C_{\rm run}(z)\subseteq\mathcal A_{\rm cert}(z)\). The second gives \(a_{\rm exec}=\kappa(z)\in\mathcal A_{\rm cert}(z)\), not necessarily \(C_{\rm run}(z)\). Hence every runtime outcome lies in \(\mathcal A_{\rm cert}(z)\). An optimizer reporting a small constraint residual is insufficient unless the residual is absorbed by a verified set erosion or the returned action is rechecked by a sound membership verifier. The recovery fallback must bypass the same failure mode that caused the task-action mechanism to fail.

### Selected architecture

The architecture choice is closed: the primary actor uses the policy-side affine generator \(T_z(u)=c(z)+G(z)\tanh u\), but \(c,G\) are outputs of the explicit certification path rather than trainable task-policy heads. Exact truncation and projection remain controls. The actor differentiates through the fixed affine map during its update, while no task gradient is allowed to modify \(c,G\) or the set certificate.

### Impact on the existing assumptions

- **Stolz et al. complement rather than weaken the safety theorem.** Their generator mechanism motivates the accepted-candidate map, but the present theorem additionally requires exactly three full-rank generators, the tanh density correction, a positive singular-value bound, and independent verification of the complete zonotope. The \(\kappa\) fallback remains separate.
- **Markgraf et al. require A9b--A9c to be explicit.** Projection can discharge accepted-candidate membership, while action aliasing and the SE-RL/SP-RL distinction show that the critic and actor cannot silently use nominal-action SAC equations. Their result does not weaken invariance if execution is exact and fallback is certified, but it weakens any unqualified claim that “projection implements constrained SAC.”
- **Yu et al. do not modify the strict assumptions unless SEditor is promoted to the safeguard.** If it replaced A9b, the guarantee would fall from robust/corridor-conditional safety to empirical low expected violation. If it is only an upstream proposal followed by M4, the original certificate remains intact and SEditor affects efficiency only.

## Revised Mathematical Definitions

The following are definitions only.

### D1. Explicit certificate state

\[
z=z^{\rm cert}
=(p,v,e,p_{\mathcal G},
\mathcal M^{\rm local},
\mathcal C^{\rm back},
\xi)
\in\mathcal Z_{\rm cert}.
\]

This is the only state used in strict theorems. The task actor receives the measurable feedforward observation \(o^{\rm task}=\Omega_{\rm task}(z,y)\); its neural features are not certificate variables.

### D2. Real dynamics

The real dynamics are (S1), including physical evolution, energy consumption, local-geometry update, return-corridor update, and task-variable update. The action argument is always the final executed acceleration \(a_{\rm exec}\).

### D3. Successor-state set

\[
\operatorname{Post}(z,a)
:=\operatorname{Post}_{\rm cert}(z,a)
\subseteq\widehat{\operatorname{Post}}_{\rm cert}(z,a).
\]

The outer envelope quantifies over all disturbances and every sound update admitted by \(\operatorname{Upd}_M,\operatorname{Upd}_C,\operatorname{Upd}_\xi\). It therefore propagates explicit sets rather than a neural latent.

### D4. Certified collision-safe set

Let \(\operatorname{Tube}_{\rm stop}(z)\) be the robust delay-and-braking tube from (S4). Let \(\underline B_{\rm geom}(z)\) be a lower bound obtained by continuous geometric verification over the complete certified cell, not by trusting \(B_\theta\). Define

\[
\mathcal S_{\rm col}^{\rm cert}
=
\left\{
z:
\operatorname{Tube}_{\rm stop}(z)\subseteq\mathcal F(z),
\quad
\underline B_{\rm geom}(z)\ge0
\right\}.
\]

Because \(\mathcal F\cap(\mathcal O\cup\mathcal U)=\varnothing\), this definition excludes both verified obstacles and unknown space. For neighborhood certification, both conditions hold uniformly on the entire certified state cell.

### D5. Certified energy-recoverable set

Let \(m_{\rm res}(z)\ge0\) denote an additional uncertainty/operational reserve beyond the terminal requirement \(e_G\). The recovery-energy field accounts for energy consumed before first entry into \(\mathcal G\), so

\[
\mathcal S_E^{\rm cert}
=\left\{
z:
e\ge
\overline R^\kappa(\bar z^{\rm cert})
+e_G+m_{\rm res}(z)
\right\}.
\]

The upper function is verified over the explicit return corridor under \(\kappa\); the learned field is only a proposal.

### D6. Charging terminal set

\[
\mathcal G
=
\left\{
z:
\begin{array}{l}
\|p-p_{\mathcal G}\|\le r_G,\quad
v\in V_G,\quad
e\ge e_G,\\
z\in\mathcal S_{\rm col}^{\rm cert},\quad
\operatorname{ChargeAdmissible}(z)
\end{array}
\right\}.
\]

Here \(e_G\) is the minimum energy needed to complete hover, descent, docking, or charging initiation. The predicate \(\operatorname{ChargeAdmissible}(z)\) means that continued safe operation or entry into charging mode is feasible. The radius \(r_G\), velocity set \(V_G\), threshold \(e_G\), and predicate \(\operatorname{ChargeAdmissible}\) remain design parameters requiring explicit certification.

### D7. Certified recovery policy

\[
\kappa:\mathcal C\subseteq\mathcal Z_{\rm cert}\to\mathcal A
\]

is a frozen deterministic controller using only \(z^{\rm cert}\). It combines verified corridor tracking, level-dependent speed limits, braking, and local obstacle avoidance. Its admissibility, robust corridor transition, collision preservation, energy recursion, and progress are continuously verified on \(\mathcal C\).

### D8. Recovery levels and corridor

Let \(K_0,\dots,K_N\) be certified state neighborhoods induced by the ordered cells of \(\mathcal C^{\rm back}\), with \(K_0=\mathcal G\). Each \(K_i\) includes explicit position, velocity, energy, local-free-space, and corridor-consistency envelopes. Define

\[
\mathcal C=\bigcup_{i=0}^N K_i,
\qquad
\ell(z)=\min\{i:z\in K_i\}.
\]

Define the recovery stopping time

\[
\tau_{\mathcal G}:=\inf\{t\ge0:z_t\in\mathcal G\}.
\]

Two progress certificates are kept separate:

- **One-step descent:** every \(z\in K_i\setminus\mathcal G\) and every \(z^+\in\operatorname{Post}(z,\kappa(z))\) satisfy \(\ell(z^+)\le i-1\).
- **Bounded-\(M\)-step descent:** for every admissible recovery path from \(z_0\in K_i\setminus\mathcal G\),
  \[
  \sigma_i:=\inf\{j\ge1:z_j\in\mathcal G\text{ or }\ell(z_j)<i\}\le M.
  \]

The second condition permits transitions within the same level before the stopped block enters a lower level. It does not imply one-step descent or one-step sup-norm contraction.

A corridor update may append or replace a cell only after its geometry is reclassified as verified free and the overlap, speed, braking, energy, and successor conditions are continuously verified. Unknown space is never appended provisionally.

### D9. Joint certified set

\[
\mathcal S_{\rm joint}^{\rm cert}
=\mathcal C\cap\mathcal S_{\rm col}^{\rm cert}\cap\mathcal S_E^{\rm cert}.
\]

### D10. Certified action set

Using a verified successor outer approximation and worst-case one-step energy cost,

\[
\mathcal A_{\rm cert}(z)=
\left\{a\in\mathcal A:
\begin{array}{l}
\widehat{\operatorname{Post}}_{\rm cert}(z,a)\subseteq
\mathcal C\cap\mathcal S_{\rm col}^{\rm cert},\\
e-\bar c(z,a)\ge
\displaystyle\sup_{z'\in\widehat{\operatorname{Post}}_{\rm cert}(z,a)}
\left[\overline R^\kappa(\bar z^{{\rm cert}\prime})+e_G+m_{\rm res}(z')\right],\\
\kappa(z')\text{ is certified for every nonterminal successor }z'
\end{array}
\right\}.
\]

The first condition includes sound local-map and return-corridor updates and therefore excludes successors entering unknown space. The last condition is not circular because certification of \(\kappa\) on \(\mathcal C\) is established before task-action certification.

### D11. Verified runtime action set

\[
\boxed{
C_{\rm run}(z)
=c(z)+G(z)[-1,1]^3
\subseteq\mathcal A_{\rm cert}(z)\cap\mathcal A,}
\qquad z\in\mathcal Z_G,
\]

\[
G(z)\in\mathbb R^{3\times3},
\qquad
\sigma_{\min}(G(z))\ge\sigma_G>0,
\qquad
C_{\rm safe\text{-}run}(z)
:=C_{\rm run}(z)\cup\{\kappa(z)\}.
\]

For \(z\notin\mathcal Z_G\), define \(C_{\rm run}(z)=\varnothing\) and \(C_{\rm safe\text{-}run}(z)=\{\kappa(z)\}\). On \(\mathcal Z_G\), the complete zonotope inclusion—not sampled actions—is verified, and its volume is \(8|\det G(z)|\ge8\sigma_G^3\). The augmented execution range includes fallback, is not assumed convex, and is not a projection target.

### D12. Runtime mechanism and executed policy

For \(z\in\mathcal Z_G\), a candidate is generated by

\[
u=m_\theta(o^{\rm task})
+s_\theta(o^{\rm task})\odot\varepsilon,
\quad
\varepsilon\sim\mathcal N(0,I_3),
\quad
\eta=\tanh u,
\quad
a_M=c(z)+G(z)\eta.
\tag{D12.1}
\]

Conditional on a valid set certificate, the candidate kernel is

\[
M_z^{\rm cand}(\cdot|u)
=\delta_{c(z)+G(z)\tanh u},
\]

and satisfies

\[
M_z^{\rm cand}(C_{\rm run}(z)|u)=1.
\]

The fail-safe kernel \(M_z^{\rm fs}\) implements (RS1): it executes the candidate only for \(z\in\mathcal Z_G\) while the complete zonotope certificate, singular-value check, deadline, and numerical checks are valid, and executes \(\kappa(z)\) otherwise. Given the Gaussian pre-squash actor \(\mu_\theta(du|o^{\rm task})\), the actual executed policy is

\[
\pi_{\rm exec}(B|z)
=\int_{\mathbb R^3}
M_z^{\rm fs}(B|u)\mu_\theta(du|o^{\rm task}).
\]

The total runtime conclusion is

\[
M_z^{\rm fs}(\mathcal A_{\rm cert}(z)|u)=1
\quad\forall z\in\mathcal S_{\rm joint}^{\rm cert},\;u\in\mathbb R^3.
\]

This definition does not assert that fallback belongs to \(C_{\rm run}\). Exact truncation and projection use different kernels and are controls rather than the primary D12 mechanism.

## Collision-Field Semantic Alternatives

| Semantics | What positive value means | Theoretical advantage | Main cost / non-claim |
|---|---|---|---|
| Dynamic braking margin | A specific braking or avoidance maneuver has enough space | Interpretable, labelable, UAV-specific | Certifies that maneuver/model, not the full viability kernel; may be very conservative |
| Discrete-time CBF | A one-step inequality can preserve a chosen superlevel set | Direct invariance and online action constraints | A learned CBF must be verified; its magnitude is not generally distance to the inevitable-collision boundary |
| Viability value | State lies in the largest set from which some policy can remain safe | Exact match to “an action always exists” | Global object; hard under partial observability and high dimension |
| Reach-avoid value | A policy can avoid failure while reaching a target/safe terminal set | Naturally connects safety and recovery | Adds target and horizon semantics; not identical to perpetual collision avoidance |
| HJ reachability value | Worst-case minimum safety margin in a differential/dynamic game | Robust, policy-independent boundary semantics | Requires known disturbance set and expensive computation/verification; learned value alone is not certified |
| Signed distance to inevitable-collision set | Geometric distance in a chosen state-space metric | Closest to the stated “dynamic margin to ICS boundary” | Computing the ICS is itself a viability/reachability problem; arbitrary signed-distance regression does not inherit invariance |

**Recommendation for later decision:** use a robust viability/HJ value as the semantic object if “boundary of the inevitable-collision set” must be claimed; use the backup corridor as the certifiable subset. A braking margin is a defensible simpler alternative if the paper narrows the claim. A CBF can certify a superlevel set, but it should not simultaneously be called signed distance or HJ value without an equivalence theorem.

## Energy-Field Semantic Alternatives

| Semantics | Bellman object | Guarantee | Theoretical cost |
|---|---|---|---|
| Expected upper bound | SSP expectation under \(\kappa\) | Expected reserve sufficiency under model distribution | Does not prevent rare exhaustion; requires properness and calibration |
| CVaR upper bound | Tail risk of cumulative recovery energy | Controls a selected tail fraction | Time consistency is nontrivial; static CVaR generally needs augmented state or nested risk recursion |
| Robust worst-case upper bound | \(\bar c+\sup_{\operatorname{Post}}R\) | Deterministic reserve guarantee under a valid uncertainty set | Most conservative; uncertainty-set validity and continuous supremum verification are demanding |

**Recommendation for later decision:** worst-case recursion for the strict theorem; expected/CVaR estimates may be reported as less conservative performance variants. This remains unselected until confirmation.

## Explicit Geometry and Generator Certification Results

### L0. Rolling-geometry and corridor-update soundness

Let Θ_s be a versioned `SensorCalibrationContract` containing finite outer bounds for position error, scan-direction error, range error, beam half-width, time synchronization error, footprint radius, grid discretization, maximum speed during synchronization delay, maximum range, evidence lifetime, and minimum independent free observations. A ray is certificate-eligible only while all fields of Θ_s are present and its timestamp, frame, validity, and range satisfy that contract. Suppose the true sensor and pose errors lie in Θ_s and the current representation satisfies

\[
\mathcal W_t
=\mathcal F_t\mathbin{\dot\cup}
\mathcal O_t\mathbin{\dot\cup}
\mathcal U_t,
\qquad
\mathcal O_{\rm true}\cap\mathcal W_t
\subseteq\mathcal O_t\cup\mathcal U_t,
\tag{L0.1}
\]

for every real obstacle set consistent with Θ_s. The update may label a closed grid cell free only when the **entire cell** lies in the guaranteed free beam tube after shrinking longitudinal and angular coverage by all Θ_s errors. It labels every cell intersecting the uncertainty-dilated hit set occupied; invalid, maximum-range, boundary-partial, insufficient-evidence, newly exposed, relocated-without-identical-world-support, and expired cells remain or revert to unknown. Each free cell stores the sensor frame, timestamp, pose interval, range interval, beam interval, calibration version, and certificate version. Suppose \(\operatorname{Upd}_C\) retains or appends a corridor cell only after proving that its uncertainty inflation lies in \(\mathcal F_{t+1}\). Then

\[
\mathcal O_{\rm true}\cap\mathcal W_t\subseteq\mathcal O_t\cup\mathcal U_t,
\qquad
\mathcal F_t\cap(\mathcal O_{\rm true}\cup\mathcal U_t)=\varnothing,
\tag{L0.2}
\]

and every retained corridor cell lies in verified free space for the evidence-validity interval.

**Proof — set containment and induction.** Full-cell free promotion implies every point of a free cell belongs to a beam region proved obstacle-free under Θ_s. Dilated hit insertion prevents a compatible hit from lying outside \(\mathcal O_t\cup\mathcal U_t\). All unsupported information is mapped to unknown, so expiry, recentering, and version changes cannot introduce a false free cell. The corridor predicate is a subset test against the resulting free union. Induction over versioned updates proves (L0.2). No learned field or finite rollout is used. \(\square\)

The implementation realizes this implication, but the physical premise that real errors lie in Θ_s is **blocked-by-calibration**. Synthetic parameter values used by tests are not aircraft calibration evidence.

### L1. Outward-rounded successor-envelope containment

Let \([p],[v],[e]\) be initial certificate intervals, \([a]\) the exact coordinate interval of the complete action zonotope, \([\Delta]=[\Delta-\epsilon_\Delta,\Delta+\epsilon_\Delta+\tau_{\rm lat}]\), and let \([w_p],[w_v],[w_a]\) be calibrated outer intervals for residual dynamics and wind/tracking. Let \(\bar c([a_{\rm all}])\) include the worst one-step energy-model underestimation. Define every arithmetic operation with directed outward rounding. The implemented affine enclosure is

\[
\begin{aligned}
[a_{\rm all}]&=[a]\oplus[w_a],\\
[p^+]&=[p]\oplus[\Delta][v]\oplus\tfrac12[\Delta]^2[a_{\rm all}]\oplus[w_p],\\
[v^+]&=[v]\oplus[\Delta][a_{\rm all}]\oplus[w_v],\\
[e^+]&=\bigl([e]\ominus[0,\bar c([a_{\rm all}])]\bigr)\cap[0,\infty).
\end{aligned}
\tag{L1.1}
\]

If the true initial state, complete action set, control-cycle duration, tracking error, disturbance, discrete-model residual, and one-step energy error lie in their stated intervals, and geometry/corridor version changes lie in the advertised update ranges, then every real one-step certificate-state successor lies in the returned envelope.

**Proof — structural induction over interval expressions.** Each primitive interval operation contains the corresponding real operation by directed rounding; Minkowski addition and interval multiplication preserve inclusion. Induction over (L1.1) gives physical-state containment. The version components are finite set ranges rather than floating-point quantities. Nonlinear residuals require a separately verified interval/Taylor/Lipschitz/branch-and-bound remainder; a point predictor is insufficient. \(\square\)

The software arithmetic is implemented. Its use as a physical outer envelope is **blocked-by-calibration** until all real bounds are supplied.

### L2. Explicit unknown-exclusion collision lemma

Assume L0. If, for every state in a certified cell,

\[
\operatorname{Tube}_{\rm stop}(z)\subseteq\mathcal F(z)
\quad\text{and}\quad
\underline B_{\rm geom}(z)\ge0,
\tag{L2.1}
\]

then every real delay, tracking, reaction, and braking trajectory represented by that tube avoids \(\mathcal O_{\rm true}\) and \(\mathcal U\).

**Proof — set containment.** By (L0.1), true obstacles in the represented window lie in \(\mathcal O\cup\mathcal U\). The partition is disjoint, so \(\mathcal F\cap(\mathcal O\cup\mathcal U)=\varnothing\). Tube containment in \(\mathcal F\) therefore excludes both true obstacles and unknown cells. The lower-envelope condition supplies the additional selected dynamic-margin premise. \(\square\)

### L3. Robust one-step collision preservation

Assume L0--L2. If a complete action set \(A_0\) has an L1 successor envelope whose position/reaction/braking tube is contained in the current verified-free union for every represented map update, then every \(a\in A_0\) preserves collision safety for one control step. This is a quantified set-containment result over \(A_0\); sampled actions, vertices, or nominal trajectories do not establish it unless a separate affine/convex completeness theorem applies.

### L4. Proof-carrying recovery transition

For every corridor state cell \(B_i\), let \(\mathsf{RCert}_i\) contain its full position/velocity/energy interval, the interval image \([\kappa](B_i)\), actuator, geometry, dynamics, energy, corridor, and policy versions, a validity interval, the verified successor enclosure, and a cryptographic digest. If the certificate verifies actuator inclusion, complete L1 successor containment in the corridor, free braking-tube containment, successor speed limits, and certificate-version equality, then every real successor under \(\kappa\) remains in the corridor while that certificate is valid.

**Proof — set containment.** The actual state and action lie in the certified cell and action interval. L1 contains the actual successor, and the stored complete-set predicates place that enclosure in the certified corridor and free geometry. Hash and version checks ensure runtime reads the same premises that were verified. \(\square\)

### L6. One-step corridor progress certificate

For the executable A8-1 profile, require each nonterminal certificate to prove

\[
\widehat{\operatorname{Post}}(B_i,[\kappa](B_i))
\subseteq \mathcal G\cup\bigcup_{j<i}B_j.
\tag{L6.1}
\]

Then \(\ell(z^+)\le\ell(z)-1\) for every represented successor. A cell for which (L6.1) cannot be proved is uncertified. The prototype does not implement A8-M and does not replace (L6.1) with point samples.

### L5. Certified recovery-action availability

Assume L4, L6, T5, and a current recovery-energy certificate linked to the same recovery-certificate hash. If the realized state lies in the certified cell and satisfies

\[
e\ge\overline R(B_i)+e_G+m_{\rm res},
\tag{L5.1}
\]

then the runtime recovery decision is authorized and \(\kappa(z)\in\mathcal A_{\rm cert}(z)\). Missing, stale, mismatched, expired, or tampered certificates do not establish L5; the software returns an emergency braking command but labels it uncertified rather than converting it into a theorem premise.

### L7. Recursive terminal-aware energy margin

Assume A2 and a valid E3 certificate. If (L5.1) holds and the executed action has a successor energy enclosure satisfying the stored cell transition, then every successor obeys

\[
e^+\ge\overline R(B_j)+e_G+m_{\rm res}
\]

for every certified successor cell \(B_j\). The result follows by subtracting the outward-rounded one-step cost upper bound from (L5.1) and applying E3. Terminal transit cost is zero; \(e_G\) is added exactly once outside \(\overline R\).

### L5a. Complete three-generator inner-set certificate

Fix a generator-enabled certified state \(z\in\mathcal Z_G\). Suppose a sound verifier proves, uniformly for every \(\eta\in[-1,1]^3\), that \(a=c(z)+G(z)\eta\) satisfies all three D10 conditions and the actuator bounds. If also \(\sigma_{\min}(G(z))\ge\sigma_G>0\), then

\[
C_{\rm run}(z)
=c(z)+G(z)[-1,1]^3
\subseteq
\mathcal A_{\rm cert}(z)\cap\mathcal A,
\tag{L5a.1}
\]

and

\[
\lambda_A(\operatorname{int}C_{\rm run}(z))
=8|\det G(z)|
\ge8\sigma_G^3.
\tag{L5a.2}
\]

If \(c,G\) are Borel measurable, the graph of \(C(z)=\operatorname{int}C_{\rm run}(z)\) is measurable.

**Proof — quantified set containment and affine volume.** The verifier premise states that every point in the affine image of the closed cube satisfies D10, which is exactly (L5a.1). Singular-value multiplication gives \(|\det G|=\prod_{i=1}^3\sigma_i(G)\ge\sigma_G^3\); the cube has volume \(8\), proving (L5a.2). Finally,

\[
(z,a)\in\operatorname{Gr}(C)
\Longleftrightarrow
\|G(z)^{-1}(a-c(z))\|_\infty<1.
\]

Matrix inversion is continuous on the nonsingular matrices, so the right-hand side is measurable when \(c,G\) are measurable. \(\square\)

L5a is the certificate-construction result. M1 below is only the mechanism-membership result; neither implies the other.

## Mechanism Validity Results

These results concern only the map from a nominal action to the executed action. They do not establish that the target action set is physically safe.

### M1. Three-generator candidate-membership theorem

Fix \(z\in\mathcal Z_G\), let \(G(z)\in\mathbb R^{3\times3}\) be full rank, and define

\[
T_z(u)=c(z)+G(z)\tanh u,
\qquad u\in\mathbb R^3.
\]

Then

\[
T_z(\mathbb R^3)
=c(z)+G(z)(-1,1)^3
=\operatorname{int}C_{\rm run}(z)
\subset C_{\rm run}(z).
\tag{M1.1}
\]

Hence, for every pre-squash actor law \(\mu_\theta(\cdot|o^{\rm task})\),

\[
\pi_{\rm cand}(B|z)
:=\int_{\mathbb R^3}
\delta_{T_z(u)}(B)\mu_\theta(du|o^{\rm task})
\]

satisfies

\[
\pi_{\rm cand}(C_{\rm run}(z)|z)=1.
\tag{M1.2}
\]

If the Gaussian actor has positive density on \(\mathbb R^3\), the topological support of \(\pi_{\rm cand}\) is the closed zonotope \(C_{\rm run}(z)\), although every finite sample lies in its interior.

**Proof.** Componentwise \(\tanh:\mathbb R^3\to(-1,1)^3\) is a bijection. An invertible affine map sends the open cube bijectively to the interior of its zonotope and sends its closure to \(C_{\rm run}\). Equations (M1.1)--(M1.2) follow by image inclusion and integration of a probability measure. This theorem proves mechanism membership only; it does not prove \(C_{\rm run}\subseteq\mathcal A_{\rm cert}\).

### M2. Exact-mask corollary

Let \(q(a|z)\) be a nominal density and assume

\[
0<Z_q(z):=\int_{C_{\rm run}(z)}q(b|z)d\lambda_A(b)<\infty.
\]

The exact distributional mask

\[
q_C(a|z)
=\frac{\mathbf1_{C_{\rm run}(z)}(a)q(a|z)}{Z_q(z)}
\]

satisfies \(q_C(C_{\rm run}(z)|z)=1\). This establishes mechanism membership. It equals the Route-A truncated Boltzmann optimizer only when \(q(a|z)\propto\exp(Q(z,a)/\alpha)\) on the actuator domain. Approximate MCMC termination or omitting the normalization gradient may retain empirical membership if every returned sample is reverified, but it does not retain the exact policy-gradient theorem.

### M3. Extension and projection-control corollary

For a deterministic mechanism \(m(z,u)\), if

\[
m(z,\mathcal U):=\{m(z,u):u\in\mathcal U\}\subseteq C_{\rm run}(z),
\]

then candidate membership follows by direct image inclusion. For generator counts \(g>3\), noninjective maps, rank-deficient matrices, or dimension changes, this image statement may remain true, but T12A's determinant density is unavailable; T10A's pushforward/induced-measure formulation is required.

For closest-point projection,

\[
P_z(u)\in\arg\min_{a\in C_{\rm run}(z)}\|a-u\|_2^2,
\]

nonempty closedness guarantees existence, and convexity guarantees uniqueness. Exact solution gives \(P_z(u)\in C_{\rm run}(z)\). Neither existence nor uniqueness proves \(C_{\rm run}(z)\subseteq\mathcal A_{\rm cert}(z)\). If the numerical solver returns only \(d(P_z(u),C_{\rm run}(z))\le\varepsilon\), membership does not follow unless the optimization is performed against a verified \(\varepsilon\)-eroded set or the candidate passes a sound post-verification test.

### M4. Fail-safe action theorem

Fix a certified state \(z\). Assume

\[
\kappa(z)\in\mathcal A_{\rm cert}(z).
\]

If \(z\in\mathcal Z_G\), additionally assume

\[
C_{\rm run}(z)\subseteq\mathcal A_{\rm cert}(z).
\]

Let the generator construction return \(c,G,a_M\), and let \(V_z(c,G,a_M)\in\{0,1\}\) be a sound verifier satisfying

\[
V_z(c,G,a)=1
\Longrightarrow
\left\{
\begin{array}{l}
\sigma_{\min}(G)\ge\sigma_G,\\
C_{\rm run}(z)=c+G[-1,1]^3
\subseteq\mathcal A_{\rm cert}(z),\\
a\in C_{\rm run}(z).
\end{array}
\right.
\]

For the fail-safe rule (RS1), with the generator branch disabled when \(z\notin\mathcal Z_G\),

\[
a_{\rm exec}=
\begin{cases}
a_M,&V_z(c,G,a_M)=1\text{ and construction returns within its certified deadline},\\
\kappa(z),&\text{otherwise},
\end{cases}
\]

the successful branch, possible only on \(\mathcal Z_G\), gives \(a_{\rm exec}\in C_{\rm run}(z)\subseteq\mathcal A_{\rm cert}(z)\). Every failure branch and every state outside \(\mathcal Z_G\) gives \(a_{\rm exec}=\kappa(z)\in\mathcal A_{\rm cert}(z)\). Therefore \(a_{\rm exec}\in\mathcal A_{\rm cert}(z)\) for every runtime outcome. The theorem does not assert \(\kappa(z)\in C_{\rm run}(z)\).

### T0. Runtime enforcement theorem

Assume L5 has certified \(\kappa(z)\in\mathcal A_{\rm cert}(z)\) throughout the corridor, L5a holds on \(\mathcal Z_G\), and A9b--A9c hold. Before task construction, an independent watchdog atomically snapshots the certificate state, computes the certified recovery decision, and stages \(\kappa(z)\) as the default command. It replaces that default only after receiving before the deadline one complete atomic bundle containing the matching snapshot/version, valid \(c,G\), full-set inclusion result, actor sample, final action, and acceptance flag. Then, for every \(z\in\mathcal S_{\rm joint}^{\rm cert}\), every \(u\in\mathbb R^3\), and every generator-construction, numerical-verification, exception, blocking, version, and timing outcome,

\[
a_{\rm exec}\in\mathcal A_{\rm cert}(z).
\]

Conditional on candidate acceptance, necessarily \(z\in\mathcal Z_G\); M1 gives \(a_{\rm exec}\in C_{\rm run}(z)\), and L5a gives \(C_{\rm run}(z)\subseteq\mathcal A_{\rm cert}(z)\). Conditional on fallback—including every \(z\notin\mathcal Z_G\)—only \(a_{\rm exec}=\kappa(z)\in\mathcal A_{\rm cert}(z)\) is claimed. The proof is M4's case split plus set containment and atomic publication. The affine generator is not used to prove that \(C_{\rm run}\) itself is safe. The Python watchdog tests this state machine; hard real-time publication remains `blocked-by-deployment-evidence` until a platform-level WCET contract is established.

### M5. SEditor non-implication

SEditor optimizes a discounted expected constraint value and an empirical violation-rate target. Such an objective does not imply

\[
E_{\rm edit}(z,u,\Delta u)\in C_{\rm run}(z)
\quad\text{for every certified }z,u,\Delta u,
\]

nor does it verify the set inclusion \(C_{\rm run}(z)\subseteq\mathcal A_{\rm cert}(z)\). A positive expected violation budget permits nonzero violation probability; even exact zero expectation under a training occupancy measure would constrain only almost-everywhere visited events, not all continuous certified states, disturbances, or distribution shifts. Therefore SEditor can become a theorem-bearing mechanism only after an independent continuous verification of its entire output image or after composition with M4. Its published result by itself remains empirical.

## Revised Theorem Statements and Proof Sketches

### Recovery-energy operators

Write

\[
\operatorname{Post}_\kappa(z):=\operatorname{Post}(z,\kappa(z)),
\qquad
\bar c_\kappa(z):=\bar c(z,\kappa(z)).
\]

The robust transit-energy value stops at the completed terminal set:

\[
R^\kappa(z)=
\begin{cases}
0,&z\in\mathcal G,\\
\bar c_\kappa(z)+\displaystyle\sup_{z^+\in\operatorname{Post}_\kappa(z)}R^\kappa(z^+),
&z\in\mathcal C\setminus\mathcal G.
\end{cases}
\tag{E1}
\]

The terminal requirement \(e_G\) is not hidden in \(R^\kappa\); it is added in D5. Although (E1) is written on the full certificate state, an energy proposal \(E_\psi(\bar z^{\rm cert})\) that excludes remaining energy is theorem-valid only when its upper residual is verified uniformly over every certified full-state fiber sharing the same explicit \(\bar z^{\rm cert}\).

### T1. Corridor-conditional joint forward invariance

Assume \(z_0=z_0^{\rm cert}\in\mathcal S_{\rm joint}^{\rm cert}\), with a sound L0 local geometry partition and a proof-carrying return corridor. Suppose L1--L3, L5, and L7 hold and every runtime action satisfies T0. Then, for every admissible disturbance and explicit geometry/corridor-update sequence and every time for which the certificates and calibrated premises remain valid,

\[
z_t\in\mathcal S_{\rm joint}^{\rm cert}
\quad\forall t\ge0.
\]

**Proof sketch — set containment and induction.** T0 gives \(a_t\in\mathcal A_{\rm cert}(z_t)\). D10 and L1 place every physical and set-update successor in the certified corridor; D4 and L3 keep the delay/braking tube in verified free space and exclude obstacle and unknown space; L7 preserves the terminal-aware energy margin. Version validity is checked before each command. These inclusions give the induction step on \(z^{\rm cert}\). Software membership alone is insufficient without L0--L1 and L7.

### T2. Recursive feasibility

Assume the hypotheses of T1 and that the proof-carrying corridor has already established L4, L6, T5, and hence L5:

\[
\kappa(z)\in\mathcal A_{\rm cert}(z)
\quad\forall z\in\mathcal S_{\rm joint}^{\rm cert}\setminus\mathcal G.
\]

Then every reachable nonterminal explicit certificate state satisfies

\[
\mathcal A_{\rm cert}(z_{t+1})\neq\varnothing,
\qquad
\kappa(z_{t+1})\in\mathcal A_{\rm cert}(z_{t+1}).
\]

**Proof sketch — set containment and induction.** T1 places \(z_{t+1}\) in the certified set. The hashed cell and energy certificates were verified corridor-wide before task-action certification, so L5 supplies \(\kappa(z_{t+1})\) directly. No generator search or task actor is invoked to prove nonemptiness.

### T3. Finite-time arrival under the two progress regimes

Let recovery start from \(z_0\in K_i\subseteq\mathcal C\), apply the explicit-state frozen controller \(\kappa\), and require every geometry/corridor update to remain in the verified update correspondence. Progress is a corridor-wide certificate premise, not a conclusion from point executions.

1. Under A8-1, \(\tau_{\mathcal G}\le i\le N\).
2. Under A8-M, \(\tau_{\mathcal G}\le Mi\le MN\).

In both cases, the terminal event is entry into the updated D6 set, including \(e\ge e_G\) and \(\operatorname{ChargeAdmissible}(z)\).

**Proof sketch — well-founded descent.** Under A8-1 the nonnegative integer rank decreases each step. Under A8-M it decreases after each stopped block of at most \(M\) steps. At most \(i\) strict rank decreases are possible before level zero.

### T4a. Unique finite recovery energy under one-step descent

Assume A8-1 has been established by L6 on every nonterminal corridor cell, \(0\le\bar c_\kappa(z)\le c_{\max}<\infty\), and \(K_0=\mathcal G\). For the finite proof-carrying partition, set \(\overline R(B_0)=0\) and recursively use each cell's full-set cost upper bound and certified lower-level successor identifiers. Then (E1) has a unique finite solution on \(\mathcal C\), satisfying

\[
0\le R^\kappa(z)\le c_{\max}\ell(z).
\]

**Proof sketch — backward induction on level.** Set \(R^\kappa=0\) on \(K_0\). If the value is uniquely defined on levels below \(i\), every successor of a state at level \(i\) lies in those lower levels, so (E1) uniquely defines the level-\(i\) value. Finite \(N\) and the cost bound give the displayed estimate. No Banach argument is used.

### T4b. Unique finite recovery energy under bounded-\(M\)-step descent

Assume A8-M, \(0\le\bar c_\kappa\le c_{\max}<\infty\), and rectangular disturbance uncertainty so that admissible recovery paths are closed under first-step decomposition and concatenation. For \(z_0=z\in K_i\setminus\mathcal G\), let \(\mathfrak P_\kappa(z)\) be the admissible recovery paths and let \(\sigma_i\le M\) be D8's path-dependent first entry time into \(\mathcal G\) or a lower level. Define the stopped block operator

\[
(\mathcal T_{\kappa,M}R)(z)
:=
\sup_{(z_j)\in\mathfrak P_\kappa(z)}
\left[
\sum_{j=0}^{\sigma_i-1}\bar c_\kappa(z_j)
+R(z_{\sigma_i})
\right],
\quad
(\mathcal T_{\kappa,M}R)(z)=0\ \text{on }\mathcal G.
\tag{E2}
\]

Then (E2) has a unique finite fixed point on \(\mathcal C\), with

\[
0\le R^\kappa(z)\le Mc_{\max}\ell(z).
\]

Moreover, the fixed point satisfies the one-step recursion (E1).

**Proof sketch — stopped \(M\)-step operator and well-founded descent.** Every continuation value in (E2) is evaluated only at \(\mathcal G\) or a strictly lower level. Backward induction therefore defines a unique block value level by level. The block cost is at most \(Mc_{\max}\). Applying the dynamic-programming principle to the first transition of each stopped path yields (E1). The ordinary one-step operator may transition within the same level and is not claimed to be a sup-norm contraction.

### T5. Conservative upper bound for recovery energy

Assume either T4a or T4b. For a reduced state \(\bar z^{\rm cert}\), define its certified full-state fiber

\[
\mathfrak F(\bar z^{\rm cert})
:=\left\{z\in\mathcal C:
\operatorname{drop}_e(z)=\bar z^{\rm cert}\right\}.
\]

Let \(\overline R^\kappa(\bar z^{\rm cert})\ge0\) be a hashed, versioned energy certificate and satisfy the following outward-rounded inequalities for every \(z\in\mathfrak F(\bar z^{\rm cert})\): terminal states require only

\[
\overline R^\kappa(\bar z^{\rm cert})\ge0
\quad(z\in\mathcal G),
\]

whereas every nonterminal fiber member requires

\[
\overline R^\kappa(\bar z^{\rm cert})
\ge
\bar c_\kappa(z)
+\sup_{z^+\in\operatorname{Post}_\kappa(z)}
\overline R^\kappa(\bar z^{{\rm cert},+})
\quad(z\in\mathcal C\setminus\mathcal G).
\tag{E3}
\]

The envelope may be set to zero at a reduced state only when every certified member of its fiber is terminal (or when the nonterminal residual is also zero). This avoids identifying two full states that share \(\bar z^{\rm cert}\) but differ in terminal eligibility because of remaining energy.

Then

\[
\overline R^\kappa(\bar z^{\rm cert})
\ge R^\kappa(z)
\ge
\sup_{(w_t)}
\sum_{t=0}^{\tau_{\mathcal G}-1}
c_{\rm real}(z_t,\kappa(z_t),w_t)
\quad\forall z\in\mathcal C.
\]

**Proof sketch — induction.** Under T4a, the implemented E3 verifier recomputes each right-hand side from the complete-cell one-step cost and linked successor certificates using outward rounding; level induction gives the bound. Under T4b, unroll (E3) only until the bounded stopping index \(\sigma_i\), producing the block inequality (E2), and then induct over levels. The second inequality follows stepwise from \(c_{\rm real}\le\bar c_\kappa\). T5 depends on progress and the residual bound, not on T6 safe arrival. Reducing a stored upper value invalidates E3 and the certificate digest.

### T6. Safe recovery arrival before energy exhaustion

Assume recovery begins in a cell carrying current recovery and energy certificates at

\[
z_0\in\mathcal S_{\rm joint}^{\rm cert}\subseteq\mathcal C,
\qquad
e_0\ge\overline R^\kappa(\bar z_0^{\rm cert})+e_G+m_{\rm res}(z_0).
\]

Assume \(\operatorname{Tube}_{\rm stop}(z_0)\subseteq\mathcal F_0\), \(\mathcal C_0^{\rm back}\) is valid, and \(\kappa\) is executable and certified throughout the corridor. If L0--L7, T3--T5, and the corresponding A8-1 or A8-M premise hold, recovery reaches \(\mathcal G\), no collision or unknown-space entry occurs before \(\tau_{\mathcal G}\), and

\[
e_{\tau_{\mathcal G}}\ge e_G
\]

(indeed at least \(e_G\) plus the preserved certified reserve when that reserve is propagated). Thus energy is not exhausted before the terminal charging condition is reached.

**Proof sketch — induction, set containment, and finite-time descent.** T1 applied to \(\kappa\) preserves collision and energy feasibility. T3 gives a finite stopping time. T5 bounds all transit energy before that time, so subtracting cumulative real cost from the initial energy leaves at least \(e_G\).

### T7. Corridor-conditional three-mode switching theorem

Assume \(z_0\in\mathcal S_{\rm joint}^{\rm cert}\), current recovery and energy certificates exist, and the platform satisfies a verified WCET contract

\[
T_{\rm sensor}+T_{\rm update}+T_\kappa+T_{\rm set}+T_{\rm actor}+T_{\rm publish}<\Delta_{\rm ctl}.
\tag{T7.1}
\]

Normal and restricted task actions use D12's generator and are executed only after T0 establishes membership in \(\mathcal A_{\rm cert}(z)\). The recovery guard triggers before the successor envelope can leave verified free geometry or invalidate \(\mathcal C^{\rm back}\). Guard evaluation, hysteresis, local-map update, mode-switch latency, computation delay, and tracking error are included in \(\widehat{\operatorname{Post}}_{\rm cert}\). The watchdog defaults to the already certified \(\kappa\); any generator-construction, inclusion, rank, numerical, exception, version, atomicity, or deadline failure leaves that fallback published.

Then the switched trajectory remains in \(\mathcal S_{\rm joint}^{\rm cert}\) for all time. If recovery mode is triggered, T6 guarantees finite-time arrival in \(\mathcal G\) before collision or energy exhaustion.

**Proof sketch — mode-wise induction and case split.** T0 covers successful task actions and fallback. T1 provides the common invariant induction step in every mode. The verified guard and (T7.1) prevent an unprotected transition between modes, atomic one-shot publication prevents a late task overwrite, and T6 handles the recovery suffix. The Python watchdog verifies transition logic only; without hardware/RTOS WCET and atomic-I/O evidence, T7 is `blocked-by-deployment-evidence`.

### T8A. Constrained soft fixed point on the certified graph

Let \(C(z)=\operatorname{int}(c(z)+G(z)[-1,1]^3)\) on \(\mathcal Z_G\). Under A9a, A10a--A10d, A11, bounded measurable reward, a transition kernel supported on \(\mathcal Z_G\), and \(0\le\gamma<1\), the Route-A operator

\[
\mathcal T_C:\mathbb B_b(\mathsf G_C)\to\mathbb B_b(\mathsf G_C)
\]

is monotone and a \(\gamma\)-contraction in the sup norm. It therefore has a unique fixed point \(Q_C^*\in\mathbb B_b(\mathsf G_C)\).

**Proof sketch — Banach contraction.** Measurability and boundedness make the operator well-defined. Log-partition over the same frozen set is 1-Lipschitz in \(Q\); expectation and multiplication by \(\gamma\) yield the contraction. Banach's theorem gives existence and uniqueness.

### T9A. Constrained soft policy improvement

Fix the same \(C\), \(P\), \(r\), \(\lambda_A\), \(\alpha\), and feasible policy class

\[
\Pi_C
=\{\pi:\pi(C(z)|z)=1,\ \pi(\cdot|z)\ll\lambda_A,\ \text{required entropy terms are finite}\}.
\]

For \(\pi\in\Pi_C\), assume the exact variational optimizer \(\pi_C^{Q^\pi}\) belongs to the same class and is realizable by the feedforward actor through T12A's full-rank generator map using A15's explicit observation. Then its entropy-regularized value satisfies

\[
Q^{\pi_C^{Q^\pi}}(z,a)\ge Q^\pi(z,a)
\quad\forall(z,a)\in\mathsf G_C.
\]

**Proof sketch — variational optimization and monotone evaluation.** The truncated Boltzmann kernel maximizes the pointwise entropy-regularized objective on the unchanged zonotope interior. Substituting that inequality into the fixed-policy Bellman operator and iterating the monotone contraction gives the value inequality. A Gaussian generator actor need not realize this optimizer; without the explicit realizability premise, T12A supplies the correct gradient but T9A's exact monotone-improvement conclusion is not claimed.

## Next-Stage Mechanism-Conditioned Route-A Theory

### Scope and proof status

This section now analyzes the selected full-rank three-generator mechanism. T10A is retained as the extension theorem for noninjective or dimension-changing mechanisms; T11A and T13A are controls; T12A is the primary actor theorem; T14A describes runtime rejection and \(\kappa\) fallback.

- T10A is provable for every measurable deterministic mechanism under disintegration.
- T11A is the ideal exact-truncation baseline.
- T12A is the selected \(u\mapsto c+G\tanh u\) implementation theorem.
- T13A proves both the valid projection formulation and the obstruction to an ordinary executed Lebesgue density.
- T14A is provable for the complete accepted-candidate/fallback mixture under an explicit mixed reference measure.

These are RL-mechanism theorems. Their physical conclusion remains only membership after L5a and T0 are invoked.

### T10A. Mechanism-induced reference-measure theorem

Fix a certified state \(z\). For this extension theorem only, let \((\mathcal U,\mathcal B_{\mathcal U},\lambda_U)\) be standard Borel with \(0<\lambda_U(\mathcal U)<\infty\), let \(T_z:\mathcal U\to C(z)\) be measurable, and define

\[
\nu_z:=(T_z)_\#\lambda_U,
\qquad
\nu_z(B)=\lambda_U(T_z^{-1}(B)).
\]

Let the nominal actor be \(\mu_\theta(du|z)=p_\theta(u|z)\lambda_U(du)\), and let

\[
q_\theta(\cdot|z):=(T_z)_\#\mu_\theta(\cdot|z)
\]

be its executed candidate law. Then \(q_\theta(\cdot|z)\ll\nu_z\). Because the spaces are standard Borel and \(\lambda_U\) is finite, there is a disintegration kernel \(\Lambda_z(du|a)\), concentrated on the fiber \(T_z^{-1}(\{a\})\), such that

\[
\lambda_U(du)=\int_{C(z)}\Lambda_z(du|a)\nu_z(da).
\tag{MRT1}
\]

The executed Radon--Nikodym density is

\[
w_\theta(a|z)
:=\frac{dq_\theta}{d\nu_z}(a|z)
=\int p_\theta(u|z)\Lambda_z(du|a)
\quad \nu_z\text{-a.e.}
\tag{MRT2}
\]

For any bounded measurable executed-action value \(Q(z,a)\), the unrestricted latent maximum-entropy problem

\[
\sup_{p\ge0,\;\int p\,d\lambda_U=1}
\left\{
\int_{\mathcal U}p(u)Q(z,T_z(u))d\lambda_U(u)
-\alpha\int_{\mathcal U}p(u)\log p(u)d\lambda_U(u)
\right\}
\tag{MRT3}
\]

is equivalent in optimal value to

\[
\sup_{w\ge0,\;\int w\,d\nu_z=1}
\left\{
\int_{C(z)}w(a)Q(z,a)d\nu_z(a)
-\alpha\int_{C(z)}w(a)\log w(a)d\nu_z(a)
\right\}.
\tag{MRT4}
\]

Its unique executed optimizer, up to \(\nu_z\)-null sets, is

\[
q_z^*(da)
=\frac{\exp(Q(z,a)/\alpha)}
{Z_{\nu}^{Q}(z)}\nu_z(da),
\qquad
Z_{\nu}^{Q}(z)
=\int_{C(z)}e^{Q(z,a)/\alpha}\nu_z(da).
\tag{MRT5}
\]

Consequently, the mechanism-conditioned soft value and Bellman operator are

\[
V_{\nu}^{Q}(z)=\alpha\log Z_{\nu}^{Q}(z),
\tag{MRT6}
\]

\[
(\mathcal T_{\nu}Q)(z,a)
=r(z,a)+\gamma\int V_{\nu}^{Q}(z')P(dz'|z,a),
\qquad (z,a)\in\mathsf G_C.
\tag{MRT7}
\]

If \(z\mapsto\nu_z\) is a measurable finite kernel and its total mass is uniformly bounded above and away from zero, then \(\mathcal T_\nu\) is a monotone \(\gamma\)-contraction on \(\mathbb B_b(\mathsf G_C)\) and has a unique fixed point.

**Proof.** If \(\nu_z(B)=0\), then \(\lambda_U(T_z^{-1}(B))=0\). Absolute continuity of \(\mu_\theta\) with respect to \(\lambda_U\) gives

\[
q_\theta(B|z)=\mu_\theta(T_z^{-1}(B)|z)=0,
\]

so \(q_\theta\ll\nu_z\). Substituting (MRT1) into \(q_\theta(B|z)\) gives

\[
q_\theta(B|z)
=\int_B\left[\int p_\theta(u|z)\Lambda_z(du|a)\right]\nu_z(da),
\]

which proves (MRT2).

For a general latent density \(p\), let \(w(a)=\int p(u)\Lambda_z(du|a)\). On fibers with \(w(a)>0\), define \(k(u|a)=p(u)/w(a)\); then \(k(\cdot|a)\) is the conditional density of the latent action relative to \(\Lambda_z(\cdot|a)\). Direct substitution yields

\[
-\int p\log p\,d\lambda_U
=-\int w\log w\,d\nu_z
-\int w(a)
\operatorname{KL}\!\left(
k(\cdot|a)\Lambda_z(\cdot|a)
\,\|\,\Lambda_z(\cdot|a)
\right)\nu_z(da).
\tag{MRT8}
\]

The KL term is nonnegative. For a fixed executed density \(w\), latent entropy is therefore maximized by the uniform-on-fiber conditional density \(k=1\), and its maximum equals the entropy of \(w\) relative to \(\nu_z\). Conversely, every \(w\) in (MRT4) is realized by \(p(u)=w(T_z(u))\), for which \(k=1\). Thus (MRT3) and (MRT4) have the same supremum. Calculus of variations applied to (MRT4) gives (MRT5)--(MRT6).

For bounded \(Q_1,Q_2\), the log-partition inequality gives

\[
|V_\nu^{Q_1}(z)-V_\nu^{Q_2}(z)|
\le \|Q_1-Q_2\|_\infty.
\]

Expectation and multiplication by \(\gamma\) prove contraction of (MRT7); monotonicity follows from monotonicity of the exponential, integral, and logarithm. Banach's theorem gives the unique fixed point. \(\square\)

**Consequence.** A deterministic mask or projection does not generally implement the \(\lambda_A\)-Boltzmann policy of T8A. It implements a Boltzmann policy relative to its geometric pushforward measure \(\nu_z\). The two statewise optimizers coincide only when \(d\nu_z/d\lambda_A\) is action-independent on \(C(z)\); exact operator equality additionally requires the reference-measure normalization to be matched.

### T11A. Ideal exact-truncation baseline

Let \(p_\theta(a|z)\) be a positive nominal density with respect to \(\lambda_A\), and freeze \(C(z)\). Define

\[
Z_\theta^C(z)=\int_{C(z)}p_\theta(b|z)d\lambda_A(b),
\qquad 0<Z_\theta^C(z)<\infty,
\]

\[
q_\theta^C(a|z)
=\frac{\mathbf1_{C(z)}(a)p_\theta(a|z)}{Z_\theta^C(z)}.
\tag{TR1}
\]

Then

\[
\log q_\theta^C(a|z)
=\log p_\theta(a|z)-\log Z_\theta^C(z)
\quad q_\theta^C\text{-a.s.}
\tag{TR2}
\]

and the exact executed-policy SAC actor loss at fixed \(z\) is

\[
L_{\rm tr}(\theta;z)
=\mathbb E_{a\sim q_\theta^C}
\left[\alpha\log q_\theta^C(a|z)-Q(z,a)\right].
\tag{TR3}
\]

Let \(s_\theta(a|z)=\nabla_\theta\log p_\theta(a|z)\). Assume differentiation under the normalizing integral is justified by an integrable derivative envelope. Then

\[
\nabla_\theta\log Z_\theta^C(z)
=\mathbb E_{q_\theta^C}[s_\theta(a|z)],
\tag{TR4}
\]

and the exact score-form actor gradient is

\[
\boxed{
\nabla_\theta L_{\rm tr}(\theta;z)
=\mathbb E_{q_\theta^C}
\left[
\left(s_\theta-\mathbb E_{q_\theta^C}s_\theta\right)
\left(\alpha\log q_\theta^C-Q\right)
\right].}
\tag{TR5}
\]

If \(\pi_C^Q\) denotes T9A's truncated Boltzmann optimizer, then

\[
L_{\rm tr}(\theta;z)
=\alpha\operatorname{KL}
\left(q_\theta^C(\cdot|z)\|\pi_C^Q(\cdot|z)\right)
-\alpha\log Z_C^Q(z).
\tag{TR6}
\]

Thus minimizing the exact loss over a realizable policy class recovers the constrained soft optimizer; truncating an arbitrary Gaussian gives only the best member of that truncated family in the forward-KL objective.

**Proof.** Equations (TR1)--(TR2) follow from conditioning the nominal measure on \(C(z)\). Differentiating \(Z_\theta^C\) under the integral yields

\[
\nabla_\theta Z_\theta^C
=\int_C p_\theta s_\theta\,d\lambda_A
=Z_\theta^C\mathbb E_{q_\theta^C}s_\theta,
\]

which proves (TR4). Therefore

\[
\nabla_\theta\log q_\theta^C
=s_\theta-\mathbb E_{q_\theta^C}s_\theta.
\]

Differentiate (TR3). The derivative of the expectation contributes the score multiplied by the integrand; differentiating \(\alpha\log q_\theta^C\) contributes \(\alpha\mathbb E_{q_\theta^C}[\nabla_\theta\log q_\theta^C]=0\). This proves (TR5). Substituting

\[
\log\pi_C^Q(a|z)=Q(z,a)/\alpha-\log Z_C^Q(z)
\]

into the KL divergence proves (TR6). \(\square\)

**Non-implication.** Omitting \(-\log Z_\theta^C(z)\), stopping its gradient, or sampling only approximately changes (TR5). Post-verification may preserve physical membership, but it does not preserve the exact actor-gradient or T9A policy-improvement claim.

### T12A. Full-rank three-generator affine-tanh policy theorem

Fix \(z\in\mathcal Z_G\), freeze \(c=c(z)\) and \(G=G(z)\), and assume

\[
G\in\mathbb R^{3\times3},
\qquad
\sigma_{\min}(G)\ge\sigma_G>0.
\tag{GEN1}
\]

Let the feedforward actor produce

\[
u=m_\theta(o^{\rm task})
+s_\theta(o^{\rm task})\odot\varepsilon,
\qquad
\varepsilon\sim\mathcal N(0,I_3),
\qquad
s_{\theta,i}>0,
\tag{GEN2}
\]

and define

\[
\eta=\tanh u,
\qquad
a=T_z(u):=c+G\eta.
\tag{GEN3}
\]

Then the following statements hold.

#### Part A: bijection and membership

\[
T_z:\mathbb R^3
\longrightarrow
C(z)=c+G(-1,1)^3
\]

is a \(C^\infty\) bijection with inverse

\[
T_z^{-1}(a)
=\operatorname{artanh}\!\left(G^{-1}(a-c)\right).
\tag{GEN4}
\]

Therefore every finite actor sample lies in \(C(z)\subset C_{\rm run}(z)\). Combined with the independently verified inclusion \(C_{\rm run}(z)\subseteq\mathcal A_{\rm cert}(z)\), this gives accepted-candidate membership in \(\mathcal A_{\rm cert}(z)\).

#### Part B: Jacobian, density, and log density

Let

\[
D(u):=\operatorname{diag}(1-\tanh^2u_1,\,
1-\tanh^2u_2,\,
1-\tanh^2u_3).
\]

The action Jacobian is

\[
J_T(z,u)
=D_uT_z(u)
=G D(u),
\tag{GEN5}
\]

and

\[
|\det J_T(z,u)|
=|\det G|
\prod_{i=1}^3(1-\eta_i^2)>0.
\tag{GEN6}
\]

Let \(\phi_\theta(u|o^{\rm task})\) be the Gaussian density in (GEN2). The accepted executed-action density with respect to three-dimensional Lebesgue measure on \(C(z)\) is

\[
\boxed{
q_\theta^{G}(a|z)
=
\frac{
\phi_\theta\!\left(
\operatorname{artanh}(G^{-1}(a-c))
\mid o^{\rm task}
\right)}
{
|\det G|
\prod_{i=1}^3
\left[
1-\left(G^{-1}(a-c)\right)_i^2
\right]
},
\quad a\in C(z).}
\tag{GEN7}
\]

Along a reparameterized sample,

\[
\boxed{
\log q_\theta^{G}(a|z)
=\log\phi_\theta(u|o^{\rm task})
-\sum_{i=1}^3\log(1-\tanh^2u_i)
-\log|\det G|.}
\tag{GEN8}
\]

The \(-\log|\det G|\) term is required even though it has zero actor-parameter derivative when \(G\) is frozen.

#### Part C: entropy and SAC actor loss

The accepted executed-policy entropy is

\[
\begin{aligned}
\mathcal H(q_\theta^G(\cdot|z))
&=\mathcal H(\phi_\theta(\cdot|o^{\rm task}))\\
&\quad+\mathbb E_{u\sim\phi_\theta}
\left[
\log|\det G|
+\sum_{i=1}^3\log(1-\tanh^2u_i)
\right].
\end{aligned}
\tag{GEN9}
\]

For a critic indexed by the actually executed action, the exact accepted-branch SAC actor loss is

\[
\boxed{
\begin{aligned}
L_G(\theta;z)
=\mathbb E_{\varepsilon}\Big[
&\alpha\log\phi_\theta(u|o^{\rm task})
-\alpha\sum_{i=1}^3\log(1-\tanh^2u_i)\\
&-\alpha\log|\det G|
-Q(z,c+G\tanh u)
\Big].
\end{aligned}}
\tag{GEN10}
\]

Using nominal Gaussian entropy, omitting the tanh correction, or omitting the affine-volume term defines a different objective.

#### Part D: exact reparameterized actor gradient

Let \(\ell_\theta=\log s_\theta\), \(u=m_\theta+e^{\ell_\theta}\odot\varepsilon\), and define

\[
r_G(z,u)
:=
2\alpha\tanh u
-D(u)G^\top
\nabla_a Q(z,c+G\tanh u).
\tag{GEN11}
\]

With \(Q,c,G,z\) frozen during the actor step,

\[
\boxed{
\nabla_{m}L_G
=\mathbb E_\varepsilon[r_G(z,u)],}
\tag{GEN12}
\]

\[
\boxed{
\nabla_{\ell}L_G
=\mathbb E_\varepsilon
\left[
-\alpha\mathbf1_3
+r_G(z,u)\odot
e^{\ell_\theta}\odot\varepsilon
\right].}
\tag{GEN13}
\]

Consequently, if \(J_m(\theta)\) and \(J_\ell(\theta)\) are the Jacobians of the feedforward actor outputs with respect to \(\theta\),

\[
\boxed{
\nabla_\theta L_G
=J_m(\theta)^\top\nabla_mL_G
+J_\ell(\theta)^\top\nabla_\ell L_G.}
\tag{GEN14}
\]

#### Proof

Componentwise \(\tanh\) is a smooth bijection from \(\mathbb R^3\) to \((-1,1)^3\). Condition (GEN1) makes the affine map \(\eta\mapsto c+G\eta\) a smooth bijection onto \(C(z)\), proving Part A.

The chain rule gives (GEN5). Since \(D(u)\) is diagonal and strictly positive at every finite \(u\),

\[
\det(GD(u))
=\det G\prod_i(1-\eta_i^2),
\]

which proves (GEN6). The multivariate change-of-variables theorem then gives (GEN7), and substitution of \(a=T_z(u)\) gives (GEN8).

For any diffeomorphism, \(\mathcal H(T_\#\phi)=\mathcal H(\phi)+\mathbb E_\phi\log|\det J_T|\). Substituting (GEN6) proves (GEN9), and inserting \(\log q_\theta^G\) into \(\mathbb E[\alpha\log q-Q]\) proves (GEN10).

To obtain the gradient, first note

\[
\nabla_u\log|\det J_T(z,u)|
=-2\tanh u.
\tag{GEN15}
\]

Differentiating (GEN10) through \(u=m+e^\ell\odot\varepsilon\) gives the generic pathwise expression

\[
\alpha\,\partial_\theta\log\phi_\theta
+\left[
\alpha\nabla_u\log\phi_\theta
+2\alpha\tanh u
-D(u)G^\top\nabla_aQ
\right]^\top\partial_\theta u.
\tag{GEN16}
\]

For a diagonal Gaussian, the explicit and pathwise Gaussian-score terms cancel in the mean derivative, yielding (GEN12). In the log-standard-deviation derivative they reduce to \(-\alpha\mathbf1_3\), yielding (GEN13). Applying the network chain rule proves (GEN14). \(\square\)

#### Optimality boundary and extensions

Because \(T_z\) is a bijection, every Lebesgue density on \(C(z)\) has a unique pullback density on \(\mathbb R^3\). Thus an unrestricted latent density class is equivalent to the feasible executed-density class of T8A. A diagonal-Gaussian latent actor is a restricted class, so T12A proves the correct loss and gradient but not automatic realization of T9A's Boltzmann optimizer.

The map is a diffeomorphism, not an isometry: tanh and a general \(G\) do not preserve Euclidean distances. No determinant formula in T12A applies to \(g>3\), noninjective generators, \(\operatorname{rank}G<3\), or dimension-changing maps. Those variants are extensions/ablations governed by T10A.

### T13A. Projection-control law, aliasing, and the two valid SAC formulations

For this control only, let \(\mathcal U\subset\mathbb R^3\) be a compact nominal-action box with finite Lebesgue reference measure, let \(\mu_\theta\ll\lambda_U\), and let \(P_z:\mathcal U\to C(z)\) be exact Euclidean projection onto a nonempty closed convex set.

#### Part A: membership does not imply a Lebesgue density

The executed law is \(q_\theta^P=(P_z)_\#\mu_\theta\), and M3 gives \(q_\theta^P(C(z)|z)=1\). Nevertheless, \(q_\theta^P\) need not be absolutely continuous with respect to Lebesgue measure on \(C(z)\). For example, let \(\mathcal U=[-2,2]\), \(C=[-1,1]\), and let \(U\) have a continuous density assigning positive probability to both \([-2,-1)\) and \((1,2]\). Then

\[
P(U)=
\begin{cases}
-1,&U<-1,\\
U,&-1\le U\le1,\\
1,&U>1.
\end{cases}
\]

The executed law has atoms of masses \(\Pr(U<-1)\) and \(\Pr(U>1)\) at the boundary. Hence an ordinary Lebesgue log density and differential entropy do not exist for the full projected law.

#### Part B: environment-side projection

Define the safeguarded nominal-action MDP by

\[
\widetilde r(z,u)=r(z,P_z(u)),
\qquad
\widetilde P(\cdot|z,u)=P(\cdot|z,P_z(u)).
\tag{PR1}
\]

Its exact nominal-entropy soft operator is

\[
(\widetilde{\mathcal T}_P\widetilde Q)(z,u)
=\widetilde r(z,u)
+\gamma\int
\alpha\log\int_{\mathcal U}
e^{\widetilde Q(z',v)/\alpha}d\lambda_U(v)
\,\widetilde P(dz'|z,u).
\tag{PR2}
\]

Under boundedness and measurability, (PR2) is a \(\gamma\)-contraction. Its unique fixed point is constant on projection fibers:

\[
P_z(u_1)=P_z(u_2)
\Longrightarrow
\widetilde Q_P^*(z,u_1)=\widetilde Q_P^*(z,u_2).
\tag{PR3}
\]

Writing this common value as \(\widehat Q_P^*(z,a)\), the identity

\[
\int_{\mathcal U}e^{\widehat Q_P(z,P_z(v))/\alpha}d\lambda_U(v)
=\int_{C(z)}e^{\widehat Q_P(z,a)/\alpha}\nu_z^P(da),
\qquad
\nu_z^P=(P_z)_\#\lambda_U,
\tag{PR4}
\]

shows that environment-side nominal SAC is equivalent to a mechanism-induced soft operator with reference measure \(\nu_z^P\), not generally to T8A with Lebesgue reference measure.

**Proof of Part B.** Equation (PR2) is the variational soft operator for the explicitly defined nominal-action MDP, so the log-sum-exp Lipschitz argument proves contraction. If \(P_z(u_1)=P_z(u_2)\), then (PR1) gives identical immediate rewards and transition kernels. The right-hand side of (PR2) is therefore identical at \(u_1,u_2\) for every input \(\widetilde Q\); in particular, its fixed point satisfies (PR3). Equation (PR4) is the defining pushforward-measure identity. \(\square\)

#### Part C: policy-side projection with executed entropy

Use T10A's disintegration for \(P_z\). Relative to \(\nu_z^P\), the executed density is

\[
w_\theta^P(a|z)
=\int p_\theta(u|z)\Lambda_z^P(du|a).
\tag{PR5}
\]

For \(w_\theta^P(a|z)>0\), define the latent posterior on the projection fiber by

\[
\mu_\theta(du|z,a)
=\frac{p_\theta(u|z)}{w_\theta^P(a|z)}
\Lambda_z^P(du|a).
\tag{PR6}
\]

Differentiation under the fiber integral gives the Fisher identity

\[
\nabla_\theta\log w_\theta^P(a|z)
=\mathbb E_{\mu_\theta(du|z,a)}
[\nabla_\theta\log p_\theta(u|z)].
\tag{PR7}
\]

For the executed-entropy loss

\[
L_{P,{\rm exec}}(\theta;z)
=\mathbb E_{a\sim q_\theta^P}
[\alpha\log w_\theta^P(a|z)-Q(z,a)],
\tag{PR8}
\]

the exact score gradient is

\[
\boxed{
\nabla_\theta L_{P,{\rm exec}}
=\mathbb E_{a\sim q_\theta^P}
\left[
\bigl(\alpha\log w_\theta^P(a|z)-Q(z,a)\bigr)
\nabla_\theta\log w_\theta^P(a|z)
\right].}
\tag{PR9}
\]

**Proof of Part C.** Differentiating (PR5) and dividing by \(w_\theta^P\) proves (PR7). Differentiating (PR8) gives a score term multiplied by its integrand and the direct derivative \(\alpha\mathbb E_q[\nabla_\theta\log w_\theta^P]\). The latter is zero because the score of a normalized measure has zero expectation, leaving (PR9). \(\square\)

Computing (PR7) requires integration over projection fibers. Backpropagating only through \(P_z(g_\theta(\varepsilon,z))\) does not compute this executed-entropy score. Without an additional valid density calculation, it optimizes a latent/nominal-entropy objective rather than (PR8).

### T14A. Complete fail-safe executed law and actor gradient

Fix \(z\in\mathcal Z_G\) and suppose the selected generator candidate is accepted with probability \(0<\beta_\theta(z)<1\). Let its conditional accepted law be

\[
q_\theta^{\rm acc}(da|z)
=w_\theta^{\rm acc}(a|z)
\lambda_A|_{C(z)}(da),
\qquad
q_\theta^{\rm acc}(\{\kappa(z)\}|z)=0,
\]

where \(w_\theta^{\rm acc}=q_\theta^G\) from T12A when the set-construction/verification acceptance event is conditionally independent of the Gaussian draw \(u\) given \(z\). If acceptance depends on \(u\), \(w_\theta^{\rm acc}\) must instead be the explicitly normalized acceptance-conditioned density. The complete fail-safe law is

\[
q_\theta^{\rm fs}(da|z)
=\beta_\theta(z)q_\theta^{\rm acc}(da|z)
+(1-\beta_\theta(z))\delta_{\kappa(z)}(da).
\tag{FS1}
\]

With mixed reference measure

\[
\rho_z:=\lambda_A|_{C(z)}+\delta_{\kappa(z)},
\]

whose components are mutually singular because Lebesgue measure assigns zero mass to the singleton \(\{\kappa(z)\}\). Its density is \(\beta_\theta w_\theta^{\rm acc}\) on the accepted component and \(1-\beta_\theta\) at the fallback atom. Define

\[
\mathsf G_{\rm fs}
:=
\{(z,a):a\in C(z)\cup\{\kappa(z)\}\},
\]

and let \(Q_{\rm fs}\) be a bounded critic on this augmented graph. This is not T8A's pure-\(C\) critic unless \(\kappa(z)\in C(z)\) and the same reference/objective are explicitly adopted. Define

\[
A_\theta(z)
:=\mathbb E_{q_\theta^{\rm acc}}
[\alpha\log w_\theta^{\rm acc}(a|z)-Q_{\rm fs}(z,a)].
\]

The exact full-policy SAC loss relative to \(\rho_z\) is

\[
\begin{aligned}
L_{\rm fs}(\theta;z)
=&\;\beta_\theta A_\theta
+\alpha\beta_\theta\log\beta_\theta\\
&+(1-\beta_\theta)
[\alpha\log(1-\beta_\theta)-Q_{\rm fs}(z,\kappa(z))].
\end{aligned}
\tag{FS2}
\]

If \(Q_{\rm fs}\) and \(\kappa\) are frozen during the actor step, then

\[
\boxed{
\nabla_\theta L_{\rm fs}
=\beta_\theta\nabla_\theta A_\theta
+(\nabla_\theta\beta_\theta)
\left[
A_\theta+Q_{\rm fs}(z,\kappa(z))
+\alpha\log\frac{\beta_\theta}{1-\beta_\theta}
\right].}
\tag{FS3}
\]

Moreover, if \(q_\theta^{\rm acc}(C_{\rm run}(z)|z)=1\), L5a holds, and \(\kappa(z)\in\mathcal A_{\rm cert}(z)\), then

\[
q_\theta^{\rm fs}(\mathcal A_{\rm cert}(z)|z)=1.
\tag{FS4}
\]

**Proof.** The two components of (FS1) are mutually singular, so their Radon--Nikodym derivatives with respect to \(\rho_z\) are \(\beta_\theta w_\theta^{\rm acc}\) and \(1-\beta_\theta\). Substitution into

\[
\int[\alpha\log(dq_\theta^{\rm fs}/d\rho_z)-Q_{\rm fs}]dq_\theta^{\rm fs}
\]

gives (FS2). Differentiating (FS2), collecting the derivative of \(\beta_\theta\), and cancelling the \(+\alpha\) and \(-\alpha\) terms gives (FS3). Equation (FS4) follows by set containment on the accepted and fallback components and addition of their probabilities. \(\square\)

The \(\nabla_\theta\beta_\theta\) term is absent from nominal SAC. Stopping this gradient, treating fallback transitions as if they came from the accepted density, or using only the nominal Gaussian log probability optimizes a different objective. At \(\beta_\theta=0\) or \(1\), (FS3) is replaced by the corresponding pure-branch loss; the interior logarithmic formula is not asserted.

### Finalized roles of T10A--T14A

1. **T12A is primary:** the full-rank three-generator affine-tanh map supplies the actual accepted-branch density, entropy, and SAC gradient.
2. **T11A is the ideal baseline:** exact truncation remains a theoretical optimum and experimental control.
3. **T13A is the projection control:** it is used to measure aliasing and objective mismatch, not as the deployed safeguard.
4. **T10A governs extensions:** \(g>3\), noninjective, rank-deficient, and dimension-changing mechanisms require induced-measure analysis.
5. **T14A governs failure:** the full law is a continuous generator branch plus the \(\kappa\) atom. If acceptance depends on \(u\), the conditional accepted density and \(\nabla_\theta\beta_\theta\) must be trained consistently.

### Proof hierarchy

The three proof branches remain separate:

1. **Certificate construction:** verified model/field/energy envelopes imply \(C_{\rm run}(z)\subseteq\mathcal A_{\rm cert}(z)\).
2. **Runtime enforcement:** candidate verification plus \(\kappa\) fallback implies \(a_{\rm exec}\in\mathcal A_{\rm cert}(z)\).
3. **RL optimization:** for one frozen measurable correspondence and one fixed model/class, \(\mathcal T_C\) has a unique fixed point and exact feasible policy improvement is monotone.
4. **Mechanism realization within the RL branch:** T12A identifies the selected generator density, entropy, and gradient; T14A adds fallback. T10A, T11A, and T13A are extension/baseline/control results. None constructs the certificate in item 1.

The RL contraction is not a physical-safety proof, and physical forward invariance is not a task-optimality proof.

## Updated Theorem Dependency Graph

| Result | Statement role | Definitions | Assumptions | Preceding results / proof framework |
|---|---|---|---|---|
| L0. Explicit-geometry update soundness | Local free/obstacle/unknown partitions and sparse return-corridor updates contain the real sensed geometry and never promote unknown space without verification | D1–D3, S1–S4 | A5, A14 | Sensor-error enclosure + finite-set update verification |
| L1. Successor-envelope containment | The true next certificate state lies in the outward-rounded envelope | D1–D3 | A1, A14 | Independent interval-containment proof plus calibrated physical/set-update bounds; no dependence on safe arrival |
| L2. Collision lower-envelope validity | \(\underline B_{\rm geom}\) and the delay/braking tube certify explicit free-space containment on each cell | D4, S4 | A3, A5 | Continuous geometric verification; the learned field is only a proposal |
| L3. One-step robust collision preservation | Any action whose certificate-state successor envelope lies in D4 has only collision-safe, non-unknown successors | D3, D4 | A1, A3, A5 | L0–L2 + set inclusion; does not presuppose task-action certification |
| L4. Recovery-policy corridor transition | A hashed complete-cell certificate proves \(\operatorname{Post}(B_i,[\kappa](B_i))\subseteq\mathcal C\) | D7, D8 | A1, A7, A14 | L0–L1 + full-cell corridor/speed/braking verification + version/hash validity |
| L6a. One-step level descent | Every nonterminal complete-cell recovery successor has lower rank | D8 | A8-1 | Proof-carrying cell successor inclusion; failed cells are excluded |
| L6b. Bounded-\(M\)-step level descent | Every recovery path enters \(\mathcal G\) or a lower level by \(\sigma_i\le M\) | D8 | A8-M | Continuous \(M\)-step tube/tree verification |
| T3. Finite-time arrival / properness | Recovery reaches updated \(\mathcal G\) within \(N\) or \(MN\) steps | D6–D8 | A8-1 or A8-M | L6a or L6b + well-founded descent |
| T4a. One-step energy uniqueness | (E1) has a unique finite solution by level induction | D5–D8, E1 | A2, A8-1, bounded cost | L6a; backward induction, independent of T6 |
| T4b. Bounded-\(M\)-step energy uniqueness | The stopped block operator (E2) has a unique finite solution and induces (E1) | D5–D8, E1–E2 | A2, A8-M, bounded cost, rectangular uncertainty | L6b; stopped-block induction, independent of T6 |
| T5. Conservative energy upper bound | The verified residual upper function dominates robust cumulative transit energy | D5, E1–E3 | A2, A4 | T4a or T4b + level/block induction; no safe-arrival premise |
| L7. One-step energy-margin preservation | The explicit one-step energy inequality preserves \(\overline R^\kappa+e_G+m_{\rm res}\) for every true successor | D2, D5 | A1, A2, A4 | L1 + T5 + direct inequality; does not presuppose T1 |
| L5. Certified recovery action exists | \(\kappa(z)\in\mathcal A_{\rm cert}(z)\) for all nonterminal certified states | D5, D7–D10 | A2–A8, A14 | L2–L4, L6a/L6b, T5, L7 + corridor-wide verification |
| L5a. Three-generator inner-set soundness | The complete \(c(z)+G(z)[-1,1]^3\) lies in \(\mathcal A_{\rm cert}(z)\cap\mathcal A\), with \(\sigma_{\min}(G)\ge\sigma_G\) | D10, D11 | A9a, A10a, A14 | L5 + continuous zonotope predecessor verification; the affine map does not prove this row |
| T0. Runtime enforcement | Every certified outcome satisfies \(a_{\rm exec}\in\mathcal A_{\rm cert}(z)\); only accepted generator candidates are asserted in \(C_{\rm run}(z)\) | D11, D12, RS1 | A9b, A9c | L5, L5a, M1, M4 + immutable snapshot and atomic watchdog case split; projection does not prove its target safe |
| T1. Joint forward invariance | \(z_0\in\mathcal S_{\rm joint}^{\rm cert}\Rightarrow z_t\in\mathcal S_{\rm joint}^{\rm cert}\) | D9–D12 | A1–A9c, A14 | L0–L3, L7, T0; induction |
| T2. Recursive feasibility | Every reachable nonterminal successor retains certified \(\kappa\) | D7, D10–D12 | A7, A9a–A9c, A14 | L5 + T1; set containment |
| T6. Safe arrival before exhaustion | Certified recovery reaches updated \(\mathcal G\) collision-free with at least \(e_G\) energy | D4–D9, E1–E3 | A1–A8, A14 | T1, T3, T5, L5, L7; finite-time descent and energy summation |
| T7. Three-mode switching safety | Certified initial state remains invariant; any recovery suffix reaches \(\mathcal G\) | D9–D12 plus verified mode guards and watchdog | A1–A11, A14, WCET contract | T0–T2, T6 + mode-wise induction, latency enclosure, atomic publication |
| T8A. Constrained soft fixed point | The zonotope-interior Route-A operator has a unique fixed point on \(\mathbb B_b(\mathsf G_C)\) | RA1–RA2 and operator | A9a, A10a–A10d, A11, \(\gamma<1\) | Measurable graph + positive volume + Banach contraction |
| T9A. Constrained soft improvement | Exact improvement is monotone inside the same frozen feasible class when realizable from explicit task input | Route-A graph, \(\Pi_C\) | A10a–A11, A15 | Variational optimizer + T8A; same \(C,P,r,\lambda_A\) |
| T10A. Mechanism-induced reference measure | Extension theorem for noninjective or dimension-changing mechanisms | MRT1–MRT8 | T10A's local finite-reference and disintegration premises | Disintegration + entropy chain rule + variational optimization + Banach contraction; independent of physical certification |
| T11A. Exact-truncation baseline | Normalized truncation has exact log density, KL objective, and score gradient | TR1–TR6 | Frozen \(C\), positive normalizer, derivative domination | Differentiation under the integral + score identity + T9A optimizer |
| T12A. Full-rank affine-tanh generator | The selected actor has membership, exact density/log density/entropy, and exact reparameterized SAC gradient | GEN1–GEN16 | A9a, A13a–A13c, A15 | M1 + change of variables + Gaussian reparameterization; physical support additionally uses L5a |
| T13A. Projection control | Projection may create singular boundary mass; environment-side SAC uses \(\nu_z^P\), while executed entropy requires fiber integration | PR1–PR9 | A13d plus T13A's local projection premises | M3 + T10A + counterexample + Banach contraction + Fisher identity |
| T14A. Fail-safe mixture | Generator acceptance and \(\kappa\) fallback form a mixed-reference law with an acceptance-probability gradient | FS1–FS4 | A13e and the stated conditional accepted density | T12A for independent acceptance, otherwise explicit conditional density; M4/L5a for support; mixture differentiation |
| T8B. Conservative-critic fixed point | Route-B shifted/shrinkage operator has unique fixed point | Weight definition | Frozen nonnegative weights/densities, \(\gamma<1\) | Pointwise stationarity + contraction |
| T8C. Safety/recovery critic semantics | Route-C critics equal their declared probability/reachability/SSP objects | Critic-specific definitions | Discount/properness/risk assumptions | Critic-specific Bellman theorem, not one generic proof |
| C1. Approximation-error guarantee | Safety survives tightened margins under bounded errors, or holds with probability \(1-\delta\) | All | A12 plus simultaneous bounds | Union/composition of L1, L2, T0, T5; model, field, and mechanism errors stated separately |

The dependency structure is:

\[
\text{verified local geometry/corridor updates and physical envelopes}
\rightarrow
\text{frozen explicit-state }\kappa
\rightarrow
\text{corridor transition and one- or }M\text{-step progress}
\rightarrow
\text{energy uniqueness and conservative upper residual}
\rightarrow
\text{certified }\kappa\text{ action}
\rightarrow
\text{verified full-rank three-generator }C_{\rm run}\subseteq\mathcal A_{\rm cert}
\rightarrow
\text{runtime enforcement in }\mathcal A_{\rm cert}
\rightarrow
\text{invariance, recursive feasibility, safe arrival, and switching}.
\]

There is no cycle: the frozen controller parameters are fixed first; L0--L1 establish independent geometric and transition envelopes; L4 and L6 generate complete-cell transition/progress certificates; T4a and T5 generate linked energy/E3 certificates without invoking safe arrival; L5 then authorizes \(\kappa\) from those proof objects. Only afterward may L5a construct a task-action inner set. T0 consumes L5/L5a and atomic watchdog evidence but does not prove either target set safe. T1--T3 and T6--T7 are downstream consequences. T12A derives the actor law without assuming physical safety; only its separate membership corollary invokes L5a. T14A invokes M4/L5a only for (FS4), so it does not feed back into T0. The RL fixed-point branch remains parallel to the physical proof.

## Executable Certification and Runtime Scheme

### Architecture decision

This stage fixes one executable reference profile without changing the theory architecture:

1. **Local geometry:** \(\mathcal M^{\rm local}\) is a fixed-capacity, world-aligned, rolling ternary grid. Each closed cell is `FREE`, `OCCUPIED`, or `UNKNOWN`, and stores its last verification step and evidence identifier.
2. **Return corridor:** \(\mathcal C^{\rm back}\) is an ordered finite list of overlapping horizontal AABBs from the charging terminal cell to the current UAV neighborhood. Each cell carries explicit position/velocity/energy intervals plus linked, hashed, versioned, expiring recovery and energy certificates. No Boolean validity input is accepted as physical evidence.
3. **Successor sets:** the reference dynamics profile uses interval arithmetic. The complete action zonotope is first enclosed by its exact coordinate interval; affine kinematics plus box-bounded disturbances then produce a conservative position, velocity, and energy envelope.
4. **Recovery progress:** the executable profile selects **one-step level descent**. This is more conservative than bounded-\(M\)-step progress but has a directly checkable predecessor condition. T4b remains valid theory but is not implemented by this prototype.
5. **Generator construction:** \(c(z)=\kappa(z)\) and \(G(z)=\operatorname{diag}(g_1,g_2,g_3)\). The diagonal subclass is still a full-rank three-generator zonotope and makes \(\sigma_{\min}(G)=\min_i g_i\) exact. A deterministic lexicographic bisection enlarges \(g_1\), then \(g_2\), then \(g_3\).
6. **Acceptance:** acceptance is a state-level result. The complete set is constructed and verified before the task actor is called. Therefore successful execution uses T12A directly. Any set-construction, inclusion, singular-value, version, numerical, actor, or deadline failure executes the already computed \(\kappa(z)\).

The reference code is isolated in `cert_runtime/` and `calibration/`. It includes a minimal Generator-SAC trainer interface but does not start a large-scale training loop or alter the existing environment stack. It remains a certifier/integration prototype, not evidence that real UAV uncertainty envelopes have been identified.

### Implementation plan and evidence gates

1. **Software-invariant gate — implemented by this prototype:** finite set types, evidence provenance, outward interval propagation, proof-object generation/invalidation, deterministic zonotope construction, independent watchdog logic, and replay separation pass focused tests. This gate does not establish a physical bound.
2. **Bound-identification gate — software interface implemented / blocked by calibration:** immutable contracts, data schemas, split validation, confidence semantics, reports, hashes, expiry, and domain guards are implemented. Real sensor, flight, battery, and terminal evidence has not been supplied.
3. **Corridor-certificate gate — implemented conditionally / blocked by calibration:** the reference verifier checks every finite AABB state cell, overlap, actuator interval, successor enclosure, stopping tube, speed, energy floor, and one-step lower-level relation using outward-rounded arithmetic. Applying those proof objects to the aircraft still requires calibrated dynamics, sensing, and energy bounds.
4. **WCET gate — blocked by deployment evidence:** benchmark worst-case grid update, corridor revalidation, \(2+3I\) complete-zonotope checks, actor inference, version recheck, and command publication on flight hardware/RTOS. `WCETContract` refuses a deployment claim without measured component bounds and an atomic publisher implementation.
5. **Closed-loop integration gate — conditionally implemented / deployment blocked:** the deterministic harness snapshots \(z^{\rm cert}\), validates versions, stages κ, closes the fixed corridor, invokes the actor only after set certification, publishes once, records tracking through an adapter, and writes \(a_{\rm exec}\) plus certificate evidence to replay. Real command-bus atomicity remains unverified.
6. **Hardware-in-the-loop evidence gate — interface implemented / evidence unresolved:** state, LiDAR, energy, tracking, timestamp, command-sink, watchdog, and log adapters exist, including replay/mock adapters and failure injection. No HIL campaign has been run.

Task-policy training starts only after Gates 2--5 establish the premises needed by the intended corridor claim. Passing Gate 1 alone supports software mechanism statements, not physical invariance.

### Conservative semantics of the finite sets

#### Ternary rolling grid

Let grid cell \(H_{ij}\subset\mathbb R^2\) be a closed world-frame AABB.

- `FREE` means a sound update has established \(H_{ij}\subseteq\mathcal F_{\rm real}\) for the validity interval of the evidence.
- `OCCUPIED` means \(H_{ij}\cap\mathcal O_{\rm inflated}\neq\varnothing\).
- `UNKNOWN` means neither inclusion has been established. Unknown is the initialization, recentering, invalid-measurement, uncovered, and expired-evidence state.
- If free and obstacle evidence conflict, `OCCUPIED` dominates.
- Grid translation is quantized to whole cells. Newly exposed cells are `UNKNOWN`; only overlapping cells preserve evidence.

This is not an occupancy-probability map. A free cell is a set-inclusion claim with provenance, while an unknown cell is never treated as low-probability free space.

#### Sparse AABB corridor chain

For ordered cells \(B_0,\ldots,B_L\), each cell is accepted only if

\[
B_i\oplus[-\varepsilon_C,\varepsilon_C]^2
\subseteq\bigcup_{H_{jk}\;\mathrm{FREE}}H_{jk},
\]

and adjacent cells have an overlap containing an axis-aligned transfer square of side \(2r_{\rm tr}\). The implementation uses AABBs because containment, inflation, overlap, and interval-successor inclusion are deterministic and bounded-time. It is sparse because it retains only the certified return chain, not all observed free space.

### Algorithm 1: conservative LiDAR-to-set update

`SensorCalibrationContract` contains

\[
(\epsilon_p,\epsilon_\theta,\epsilon_r,\beta,
\epsilon_t,r_{\rm body},\epsilon_{\rm grid},d_{\max},v_{\max},
\tau_{\rm age},n_{\rm free},v_{\rm cal}),
\]

respectively bounding pose translation, attitude/direction, range, beam half-width, synchronization time, footprint, cell discretization, maximum range, motion during synchronization, evidence age, required independent free observations, and calibration version. Any missing member makes the contract `blocked-by-calibration` and prevents free-space certification.

For a valid hit beam with measured range \(d_i\), define the guaranteed free length and angular half-width

\[
d_i^{\rm free}
=\max\{0,d_i-\epsilon_p-\epsilon_r-v_{\max}\epsilon_t-r_{\rm body}-\epsilon_{\rm grid}\},
\qquad
\beta_i^{\rm free}=\max\{0,\beta-\epsilon_\theta\}.
\]

A cell becomes free only if every one of its corners lies strictly inside this shortened angular sector and the required number of distinct `(sensor_frame,timestamp)` observations agree. Maximum-range/no-hit rays, invalid rays, cells only partially covered, and window-boundary cells certify no free space. For a hit, every cell intersecting the endpoint disk with radius

\[
r_i^{\rm obs}=\epsilon_p+\epsilon_r+r_{\rm body}+\epsilon_{\rm grid}
+d_i\sin(\epsilon_\theta+\beta)
\]

becomes occupied. `OCCUPIED` dominates `FREE`. Every accepted free cell stores `EvidenceProvenance(sensor_frame, timestamp, pose_interval, range_interval, beam_interval, calibration_version, certificate_version)`. Evidence older than \(\tau_{\rm age}\) expires to unknown.

```text
UPDATE_LOCAL_GEOMETRY(pose, rays, previous_grid):
    snap rolling origin to whole grid cells
    preserve only old/new overlap; initialize new cells UNKNOWN
    free_candidates <- empty
    obstacle_candidates <- empty
    reject the update if calibration is incomplete
    for each time-consistent valid hit ray in fixed world direction:
        compute shortened guaranteed-free sector
        add cell only if the complete closed cell lies in that sector
        add every cell intersecting the uncertainty-dilated endpoint disk
    mark obstacle_candidates OCCUPIED
    promote sufficiently evidenced free_candidates minus obstacle_candidates
    expire stale FREE cells to UNKNOWN
    increment geometry version
```

Whole-cell recentering preserves evidence only for identical world cells; fractional relocation or unprovable frame changes revert affected cells to unknown. The prototype enumerates all cells for clarity, with cost \(O(BWH)\) for \(B\) beams and a \(W\times H\) grid. A production rasterizer may restrict enumeration to conservative beam/hit bounding boxes but may not weaken the full-cell predicate.

### Algorithm 2: return-corridor lifecycle

**Geometry creation.** Starting at the parameterized terminal cell, create an ordered list only if every cell's horizontal projection contains its declared position interval, its velocity interval respects the cell speed limit, every inflated cell is covered by current free cells, and every adjacent overlap contains the transfer square. Creation invalidates all old proof objects.

**Proof installation.** An epoch-level verifier must certify **all** cells before the corridor becomes recovery-authorizing. It first installs `RecoveryCellCertificate` objects, then solves and installs linked `RecoveryEnergyCertificate` objects. Partial dictionaries, version mismatch, or hash mismatch are rejected atomically.

**Extension.** Append exactly one higher-level geometry cell after verifying the same free-space, state-bound, speed, and overlap predicates. Unknown cells cannot be appended. Extension invalidates all recovery and energy certificates until the complete chain is reverified.

**Failure detection.** Recheck every corridor cell against the latest geometry. The first failed cell disconnects its entire outward suffix from the station.

**Deletion.** If the UAV is already inside the remaining valid prefix, delete the invalid suffix atomically, increment the geometry/corridor versions, and invalidate every proof object tied to the previous chain.

**Safe migration.** If the UAV is in the invalid suffix, mark the suffix invalid but retain it until the UAV has entered the last valid predecessor. The only admissible target is the previously verified overlap/predecessor. If the new observation invalidates the current braking tube or the migration connection itself, the corridor theorem has lost a premise: execute emergency braking, mark the certificate invalid, and report `unresolved-corridor-loss`; do not claim safe return.

```text
REVALIDATE_CORRIDOR(grid, corridor, current_position):
    j <- first cell whose inflated AABB is not fully FREE
    if no j: bump version, invalidate certificates, return REVERIFY-REQUIRED
    if current_position lies in valid prefix [0, j):
        delete suffix [j, L]; bump version; return VALID-SHRUNK
    mark suffix [j, L] invalid
    if predecessor j-1 and its migration overlap remain certified:
        return MIGRATE-UNDER-KAPPA
    return CERTIFICATE-LOST-EMERGENCY-BRAKE
```

### Algorithm 3: certificate-state successor envelope

The independent scalar type `Interval([x^-,x^+])` rejects NaN/Inf and implements every primitive operation with

\[
\operatorname{down}(x)=\operatorname{nextafter}(x,-\infty),
\qquad
\operatorname{up}(x)=\operatorname{nextafter}(x,+\infty).
\]

Addition, subtraction, multiplication, scalar multiplication, inflation, saturation, and intersection are composed only through this type. For initial intervals \([p],[v],[e]\), complete zonotope coordinate enclosure \([a]\), tracking and wind interval \([w_a]\), residuals \([w_p],[w_v],[w_e]\), and

\[
[\Delta]=[\Delta_0-\epsilon_\Delta,
\Delta_0+\epsilon_\Delta+\tau_{\rm sense/compute/switch}],
\]

the prototype computes

\[
\begin{aligned}
[a_{\rm all}]&=[a]\oplus[w_a],\\
[p^+]&=[p]\oplus[\Delta][v]\oplus\tfrac12[\Delta]^2[a_{\rm all}]\oplus[w_p],\\
[v^+]&=[v]\oplus[\Delta][a_{\rm all}]\oplus[w_v].
\end{aligned}
\]

With nonnegative calibrated energy coefficients \(q_i\), fixed/uncertainty cost \(c_f,c_u\), and worst energy-consumption underestimation \(\epsilon_e\),

\[
\bar c_{\rm step}
=c_f+c_u+\epsilon_e+\sum_iq_i
\max\{|a_i^{\rm all,-}|,|a_i^{\rm all,+}|\},
\qquad
[e^+]=\left([e]\ominus[0,\bar c_{\rm step}]\right)\cap[0,\infty).
\]

The floating-point numerical budget is not an informal scalar added afterward: each elementary endpoint is enlarged by one representable number through `nextafter`, so the accumulated budget is the compositional interval width generated by the expression tree. The envelope also carries dynamics/energy-bound versions, geometry/corridor version ranges \([v,v+1]\), and a mandatory revalidation flag. This records that perception/corridor updates are part of the certificate-state transition. It does not by itself prove that every possible physical or set update lies in the modeled correspondence; that remains an A1/L0 calibration and verification obligation.

For nonlinear residual dynamics, the affine interval formula must be replaced or enlarged by a verified interval, zonotope, Taylor-model, Lipschitz remainder, or complete branch-and-bound enclosure. A neural point prediction is not accepted. Corner and random-interior tests detect implementation errors only; affine interval inclusion, not those tests, is the completeness argument.

### Algorithm 4: executable frozen recovery policy

The controller uses only \(z^{\rm cert}\). At level \(i>0\), it targets the center of \(B_{i-1}\):

\[
\kappa(z)=\operatorname{clip}_{\mathcal A}
\left(K_p(p_{i-1}^{\rm ctr}-p)-K_vv\right).
\]

At runtime, `certified_action` first locates the realized state in a corridor **state interval**, validates the recovery and energy certificate hashes, versions, validity times, and bound versions, and checks that the point controller output lies in the preverified interval image \([\kappa](B_i)\). Missing or stale evidence returns braking with `certified=False`; that operational command is not silently promoted to L5.

At corridor-certification time, `CorridorRecoveryVerifier` evaluates each entire state cell. It computes \([\kappa](B_i)\) by monotone interval saturation, checks actuator bounds, propagates that complete interval through Algorithm 3, verifies the full braking tube against `FREE`, checks successor speed and energy lower bounds, and requires

\[
\widehat{\operatorname{Post}}(B_i,[\kappa](B_i))
\subseteq
\begin{cases}
\mathcal G,&i=0,\\
\bigcup_{j<i}B_j,&i>0.
\end{cases}
\]

It then emits a `RecoveryCellCertificate` containing cell/level/state intervals, predecessor/successor identifiers, all parameter and data versions, the action interval, verified successor enclosure, progress result, transit-cost bound, validity interval, and digest. If any cell fails, no certificates are installed. This implements only A8-1; A8-M remains theory-only until a stopped-block continuous verifier exists.

`TerminalCondition` now parameterizes the terminal position set, altitude interval, terminal velocity interval, minimum energy \(e_G\), and Boolean continuation permissions for hover, descent, and docking. Terminal transit energy is zero; at least one continuation mode must be admissible.

### Algorithm 4b: recovery-energy certificate

After all recovery cells pass, `RecoveryEnergySolver` processes levels in increasing order:

\[
\overline R(B_0)=0,
\qquad
\overline R(B_i)=
\operatorname{up}\!\left(
\bar c_i+\max_{j\in\operatorname{Succ}(i)}\overline R(B_j)
\right).
\]

Here \(\bar c_i\) is the full-cell, full-action-interval upper cost stored by the linked recovery certificate. The E3 verifier independently recomputes

\[
\rho_i=
\operatorname{down}\!\left(
\overline R(B_i)-
\operatorname{up}\!\left(\bar c_i+max_{j\in\operatorname{Succ}(i)}\overline R(B_j)\right)
\right)
\]

and requires \(\rho_i\ge0\). Each result stores the recovery-certificate hash, energy-bound/corridor versions, validity interval, and its own digest. Runtime requires \(e\ge\overline R+e_G+m_{\rm res}\). The solver reports `blocked-by-calibration` when the physical energy contract is incomplete.

### Algorithm 5: deterministic three-generator inner zonotope

The reference constructor uses \(c=\kappa(z)\) and diagonal \(G\). Let

\[
g_i^{\max}=\min\{c_i-a_i^{\min},a_i^{\max}-c_i\}.
\]

If any \(g_i^{\max}<\sigma_G\), no full-rank zonotope exists under this profile and runtime falls back. Otherwise initialize \(g_i=\sigma_G\), verify the complete minimum zonotope, then maximize each coordinate in fixed order \(x,y,z\) by a fixed-count bisection. The deterministic order is the tie-breaking rule.

For every trial zonotope, verification performs:

1. a certified lower bound on \(\sigma_{\min}(G)\); for the selected diagonal \(G\), it is exactly \(\min_i g_i\), and no failure path lowers \(\sigma_G\);
2. an outward upper bound on the condition number, exact for the diagonal profile;
3. outward row-sum action bounds \(c_i\pm\sum_j|G_{ij}|\) for actuator inclusion;
4. interval propagation of that complete action box through Algorithm 3;
5. full-cell coverage of the entire successor stopping AABB;
6. containment of the entire successor position/velocity/energy envelope in a cell carrying matching recovery and E3 energy certificates;
7. recovery-feasibility, energy-reserve, hash, version, validity-time, finite-number, and deadline checks.

This is a complete-set sufficient check for the implemented affine-box model. It does not rely on sampled actions, sampled trajectories, or vertex-only dynamics evaluation. The action interval is an exact coordinate enclosure of \(c+G[-1,1]^3\), and L1's affine interval propagation contains every zonotope successor. Correlation loss makes the result conservative, not unsound. Current collision/corridor predicates are full-box subset tests. Any future nonmonotone or nonconvex constraint requires a sound interval subdivision/branch-and-bound verifier; until then that extension is `unresolved`.

The constructor costs at most \(2+3I\) complete verifier calls for \(I\) bisection iterations: one minimum-set check, \(3I\) lexicographic trials, and one final check. A verifier costs \(O(N_{\rm stop}+L)\), where \(N_{\rm stop}\) is the number of grid cells intersecting the stopping AABB and \(L\) is corridor length. Thus construction is \(O(I(N_{\rm stop}+L))\), excluding map update. If the minimum full-rank set fails, it returns `NO_GENERATOR_SET`; rank-deficient T12A execution is forbidden.

### Algorithm 6: runtime acceptance and fallback

```text
WATCHDOG_STEP(state, task_observation):
    snapshot <- ATOMIC_CERTIFICATE_SNAPSHOT(state)
    recovery <- KAPPA_CERTIFIED_ACTION(snapshot)       # no task dependency
    if recovery is not certified:
        publish_once(recovery.brake, UNCERTIFIED_EMERGENCY)
        return                                         # no strict claim
    publisher.default <- recovery.action               # stage kappa first
    start task worker(snapshot, task_observation):
        cert <- CONSTRUCT_AND_VERIFY_COMPLETE_ZONOTOPE(snapshot, recovery)
        u <- FEEDFORWARD_TASK_ACTOR(task_observation) only if cert is valid
        eta <- tanh(u); candidate <- c + G eta
        bundle <- (snapshot, matching version, c, G, inclusion hash,
                   u, eta, candidate, atomic acceptance=true)
        atomically submit complete bundle
    until deadline:
        if complete bundle received and snapshot/version still match:
            publish_once(bundle.candidate, ACCEPTED)
            return
        if exception, worker death, version change, or incomplete bundle:
            break
    publish_once(publisher.default, FALLBACK_REASON)
```

The state-level set construction decision is independent of \(u\). Therefore, conditional on a valid state certificate and finite actor output, the accepted policy is exactly T12A's affine-tanh pushforward. The prototype `SimulatedWatchdog` uses a daemon worker so a blocked task solver cannot prevent fallback publication, immutable snapshots to reject stale candidates, and `AtomicCommandPublisher` to permit exactly one command. It is a state-machine simulation, not RTOS scheduling or WCET evidence. Actor/nonfinite failures remain deterministic fallback events; the implementation does not fabricate a differentiable \(\beta_\theta\).

### Module interfaces and data flow

| Module | Primary interface | Inputs | Outputs / invariant |
|---|---|---|---|
| `cert_runtime/contracts.py` | `SensorCalibrationContract`, `WCETContract` | calibrated sensor/error bounds or measured latency evidence | explicit `implemented`, `blocked-by-calibration`, or `blocked-by-deployment-evidence` gate |
| `cert_runtime/interval.py` | `Interval` | finite endpoints | independently outward-rounded scalar interval operations |
| `cert_runtime/geometry.py` | `RollingLocalGeometry.update_lidar` | pose, frame/time-stamped rays, calibrated `SensorBounds` | versioned ternary grid; each free cell carries complete provenance and expiry |
| `cert_runtime/certificates.py` | proof-object dataclasses and hashes | state cells, versions, enclosures, validity intervals | immutable recovery/energy/terminal certificate structures |
| `cert_runtime/corridor.py` | lifecycle and certificate installation | grid, AABB state cells, proof dictionaries | ordered geometry chain and atomic linked-certificate installation/invalidation |
| `cert_runtime/state.py` | `snapshot` | \(p,v,e,p_G\), grid, corridor, explicit task state | immutable snapshot and `(geometry,corridor,certificate_epoch)` version |
| `cert_runtime/envelope.py` | `propagate_zonotope`, `propagate_interval_state` | interval state, complete zonotope/action interval, bound versions | outward position, velocity, energy, and update-version envelope |
| `cert_runtime/recovery.py` | `CorridorRecoveryVerifier.verify`, `certified_action` | complete corridor cells and current proof objects | all-cell recovery certificates or failure; runtime authority never comes from a Boolean |
| `cert_runtime/energy.py` | `solve`, `verify_residuals` | certified recovery DAG and calibrated energy bounds | T4a upper values and linked E3 certificates |
| `cert_runtime/zonotope.py` | `construct`, `verify_complete` | state, \(\kappa(z)\), deadline | verified full-rank zonotope or explicit failure reason |
| `cert_runtime/actor.py` | `action_and_log_density` | task observation, detached \(c,G\) | Torch action and T12A log density including tanh and \(-\log|\det G|\) corrections |
| `cert_runtime/watchdog.py` | `SimulatedWatchdog.run` | immutable snapshot, default \(\kappa\), asynchronous producer | complete atomic candidate or one-shot default fallback |
| `cert_runtime/runtime.py` | `RuntimeCertifier.step`, `prepare_candidate_bundle` | state, task observation | accepted candidate or immediate certified \(\kappa\), plus full replay record |
| `tests/test_cert_runtime.py` | `unittest` suite | deterministic synthetic fixtures | software evidence for geometry, intervals, certificates, energy, generator, watchdog, and replay logic |
| `tests/test_actor_torch.py` | gradient/formula tests | Torch, if installed | exact stable Jacobian/log-determinant and frozen-\(c,G\) checks; otherwise explicit skip |

The data flow is

\[
\text{LiDAR/pose}
\to\mathcal M^{\rm local}
\to\mathcal C^{\rm back}
\to z^{\rm cert}
\to(\kappa,\widehat{\operatorname{Post}},c,G)
\to\text{state-level certificate}
\to\text{actor map or fallback}
\to a_{\rm exec}\to\text{replay}.
\]

The task observation branches only after the state-level set certificate succeeds. It never supplies geometry to the certificate path through a learned feature.

### Certificate lifecycle and invalidation

| Certificate object | Generation | Runtime verification | Version binding | Invalidation rule |
|---|---|---|---|---|
| `EvidenceProvenance` | complete calibrated LiDAR free-tube or external continuous proof | free-cell query requires nonempty live evidence | sensor calibration, geometry certificate version, frame/time | age expiry, nonidentical recentering, conflicting/static occupied evidence, geometry replacement |
| `RecoveryCellCertificate` | all-state-cell \([\kappa](B_i)\) propagation and A8-1 verification | recompute digest; check state membership, policy/dynamics/energy/geometry/corridor versions and validity time | cell, geometry, corridor, \(\kappa\), dynamics, energy | any bound/policy/geometry/corridor change, expiry, digest mismatch, failed cell re-verification |
| `RecoveryEnergyCertificate` | T4a backward recursion after all recovery certificates exist | recompute digest and E3 residual; check linked recovery hash and energy/corridor versions | recovery digest, energy model, corridor | linked recovery change, energy-bound/corridor change, expiry, negative recomputed residual |
| `ZonotopeCertificate` | deterministic \(2+3I\) full-set verification around certified \(\kappa\) | rank, conditioning, full inclusion digest, certificate version, finite values, deadline | complete certificate-state version and recovery hash | state/version/hash change, rank/condition/numerical failure, deadline, any failed predicate |
| `CertificateStateSnapshot` | atomic copy before task construction | equality of values, geometry digest, corridor digest, and certificate epoch | geometry/corridor/certificate epoch | any mutation of the proof-bearing state |
| `CandidateBundle` | only after set certification and actor return | actor sample finite; \(\eta=\tanh u\); final action equals \(c+G\eta\); complete inclusion object and snapshot match | snapshot and zonotope certificate | incomplete/mismatched fields, exception, worker death, timeout, version change |
| `PublishedCommand` | watchdog one-shot publication | single-assignment register | snapshot certificate version | no overwrite; next control cycle creates a new publisher |

Cryptographic digests provide software integrity/version linkage, not adversarial security or physical truth. They cannot repair an invalid calibration contract.

### Complexity budget

For \(B\) rays and a \(W\times H\) grid, the clear reference LiDAR updater is \(O(BWH)\). Corridor geometry validation is \(O(LN_C)\), where \(N_C\) is the number of grid cells touched by an inflated corridor AABB. Complete-cell recovery verification is \(O(L(N_{\rm stop}+L))\) with the current linear containing-cell search. T4a recursion and E3 verification are \(O(L+E)\) for \(E\) certified lower-level edges. Generator construction makes at most \(2+3I\) verifier calls and costs \(O(I(N_{\rm stop}+L))\). Runtime affine-tanh mapping is \(O(1)\) in fixed dimension. These asymptotic bounds are not WCET measurements.

### Numerical tolerances and deadlines

The prototype exposes all safety-relevant values through versioned dataclasses rather than hiding constants:

- `SensorCalibrationContract`/`SensorBounds`: pose, attitude, range, beam, synchronization, footprint, discretization, maximum range/speed, evidence age/count, and calibration version;
- `DynamicsBounds`: control period/error, sensing/compute/switch latency, initial-state uncertainty, residual position/velocity, action tracking, wind, calibration flag, and version;
- `EnergyBounds`: full-action analytical upper-cost coefficients, additive error, calibration flag, and version;
- `CertificateConfig`: actuator bounds, \(\sigma_G\), condition-number ceiling, \(e_G\), reserve, braking, latency, geometry margin, tolerance, deadline, and bisection count;
- `RecoveryConfig` and `TerminalCondition`: controller version/gains/limits and terminal position, velocity, energy, and continuation-mode predicates;
- `WCETContract`: sensor, update, recovery, set-construction, actor, and publication bounds plus deployment evidence identifier.

The tests use synthetic tolerances, \(\sigma_G\), deadlines, and calibrated flags only as deterministic fixtures. They are not flight values. `Interval` applies IEEE-754 `math.nextafter` outward rounding after each primitive operation, rejects nonfinite endpoints, and is the only arithmetic authority for strict scalar intervals. This improves software enclosure discipline but is not a formal verification of the Python interpreter, libm, processor, or compiler. A flight implementation needs platform-qualified directed rounding or a separately verified numerical-error contract.

`SimulatedWatchdog` stages \(\kappa(z)\), runs task construction in an independent daemon worker, accepts only a complete matching atomic bundle before the deadline, and otherwise publishes the default exactly once. It demonstrates fail-closed state-machine semantics even when a worker blocks, but cannot prove scheduler preemption, actuator-bus atomicity, or hard deadline compliance. Those premises remain `blocked-by-deployment-evidence`.

### Minimal prototype and tests

The prototype contains no environment training loop. `cert_runtime/actor.py` and `cert_runtime/trainer.py` are Torch-dependent; the certifier, calibration pipeline, proof manifest, watchdog, and most tests use the Python standard library so their logic remains independently auditable.

The focused test suite verifies the mandatory software properties: unknown-by-default geometry, invalid/max-range/partial-ray exclusion, error dilation, evidence provenance/expiry/recentering/version invalidation; outward arithmetic and full-action/state/error containment; all-cell recovery progress and certificate hashing; T4a order and E3 tamper detection; full-zonotope actuator/rank/conditioning/collision/corridor/energy/version/deadline checks; blocked-solver watchdog fallback and one-shot publication; and nominal/candidate/executed/replay separation. Random interior checks are explicitly implementation diagnostics, not completeness proofs.

The runnable command is

```bash
.venv/bin/python -m unittest discover -v -s tests -p 'test_*.py'
```

Previous package baseline recorded on 2026-08-05 before the calibration/closure/trainer additions:

```text
Ran 42 tests in 0.623s
OK (skipped=2)
```

- Passed: 40.
- Skipped: 2, both in `tests/test_actor_torch.py`, because Torch was not installed in the baseline interpreter.
- Failed: 0.
- Not run: real sensor calibration, hardware/RTOS WCET, actuator-bus atomicity, HIL disturbance containment, and real energy-model validation.
- The skipped Torch checks were not counted as passed. They were executed later in this round; see **Current test evidence**.

## Certificate Closure Audit

This is the strict **Certificate Closure Audit**. `implemented` means the finite software mechanism and its stated conditional implication exist; it never means that an uncalibrated physical premise is true. `blocked-by-calibration` requires sensor/dynamics/energy evidence. `blocked-by-deployment-evidence` requires platform WCET/atomic-I/O evidence. `theory-only` and `control/baseline-only` are not runtime claims.

#### Assumptions A1--A15

| Item | Current Mathematical Premise | Current Code Object | Current Evidence | Missing Content | Can Close This Round? | Status |
|---|---|---|---|---|---|---|
| A1 | Real one-step certificate successor lies in `Post` | dynamics/tracking contracts, `Interval`, `SuccessorEnvelopeBuilder` | split/exceedance plus corner/interior/combined-error tests | real aircraft state, wind, tracking, residual, latency, and update evidence | No: physical data required | **blocked-by-calibration** |
| A2 | Real one-step energy is bounded by \(\bar c\) | energy contract and `EnergyBounds.cost_upper` | underestimation report plus outward full-action cost tests | identified propulsion/avionics/environment bounds | No | **blocked-by-calibration** |
| A3 | Collision theorem uses explicit verified geometry | ternary grid, stopping-box subset predicate | full-cell/provenance/unknown tests | real sensor error enclosure and footprint validation | No | **blocked-by-calibration** |
| A4 | \(\overline R\) satisfies corridor-wide E3 | `RecoveryEnergySolver`, energy certificates | level-order and tamper/E3 tests | calibrated A2 and real corridor cells | Software closed; physical no | **blocked-by-calibration** |
| A5 | Complete delay/reaction/braking tube stays in known free space | sensor contract, geometry update, recovery/generator verifiers | timing/dilation/stale tests | calibrated sensing horizon, brake and latency bounds | No | **blocked-by-calibration** |
| A6 | Terminal set includes position, velocity, \(e_G\), and evidenced continuation | terminal contract and `TerminalCondition` | predicate, evidence-required, version, and separate-energy tests | real hover/descent/docking/charging evidence | Interface yes; physical no | **blocked-by-calibration** |
| A7 | Frozen \(\kappa\) is admissible on every corridor state cell | `FrozenRecoveryPolicy`, `CorridorRecoveryVerifier`, manifest | complete-cell action/successor and all-cell closure tests | calibrated plant and deployed corridor proof | Software conditional only | **blocked-by-calibration** |
| A8-1 | Every nonterminal recovery cell reaches a lower level in one step | recovery cell verifier/progress field | failing cell is rejected; all installed cells have lower successors | real calibrated corridor certificate | Software conditional only | **blocked-by-calibration** |
| A8-M | Stopped block reaches a lower level within \(M\) | none | none | stopped-block continuous verifier | No; intentionally excluded | **theory-only** |
| A9a | Full-rank diagonal zonotope is wholly inside \(\mathcal A_{\rm cert}\) | `ZonotopeConstructor.verify_complete` | full-action interval, rank, conditioning, reserve tests | physical validity of L0/L1/L5 | Conditional mechanism closed | **partial** |
| A9b | Only complete timely matching candidates execute; otherwise \(\kappa\) | runtime, watchdog trace, one-shot publisher, WCET harness | timeout/block/exception/stale/mutation/publish failure injections | RTOS WCET and atomic command bus | No | **blocked-by-deployment-evidence** |
| A9c | Critic/replay use executed action, tracking error in `Post` | `ReplayRecord`, `GeneratorSACTrainer`, tracking contract | action-separation, critic-input, replay-epoch, and tracking tests | calibrated tracking bound and deployed measurement linkage | Software integrated; physical no | **partial** |
| A10a | \(c,G\) and generator graph are measurable | finite versioned encoding plus fixed-iteration lexicographic constructor | deterministic tie-breaking and finite-arithmetic source audit | kernel-level formalization beyond the affine/AABB profile | Selected profile conditionally closed | **partial** |
| A10b | Accepted sets have positive Lebesgue measure | sigma/rank checks, diagonal `Zonotope3` | no-rank-reduction and volume-related checks | uniform existence over claimed certified domain | No | **partial** |
| A10c | Bounded reward and measurable kernel remain on generator domain | no RL kernel in prototype | none | formal safeguarded MDP kernel | No | **unresolved** |
| A10d | partition/value/policy kernels are measurable | theorem definitions only | formula audit | product-space implementation/kernel measurability | No | **theory-only** |
| A11 | \(C,c,G,P,r\) frozen within policy-improvement proof | detached actor map and `CertificateEpoch` | epoch-freeze and stale-replay rejection tests | environment kernel/reward freeze outside this minimal trainer | Trainer portion closed | **partial** |
| A12 | Probabilistic bounds are simultaneous over cells/time | confidence semantics and simultaneous-bound interface | semantic rejection tests | real simultaneous coverage construction over claimed horizon | Interface yes; evidence no | **blocked-by-calibration** |
| A13a | Gaussian pre-squash actor and affine-tanh map | `actor.py` | CPU-Torch action, density, and formula tests executed | large-scale optimization not required for this premise | Yes for selected actor | **implemented** |
| A13b | Differentiation valid with frozen \(c,G\) | detach logic and Generator-SAC actor update | autograd/finite-difference and no-gradient-into-\(c,G\) tests executed | analytical integrability over the learned parameter sequence | Implementation yes; theorem premise partial | **partial** |
| A13c | T12A applies only to invertible 3-by-3 \(G\) | rank/condition guard | sigma/no-generator tests | none for selected finite mechanism | Yes | **implemented** |
| A13d | truncation/projection are controls | no primary-path dependency | architecture/source audit | separate control implementations if experiments need them | Not required now | **control/baseline-only** |
| A13e | fallback mixture has explicit acceptance law | state-level 0/1 branch and replay | fallback/replay tests | learned differentiable \(\beta_\theta\), intentionally absent | No claim needed | **partial** |
| A14 | finite explicit set updates are versioned and sound | grid/corridor/proof objects, evidence hashes, invalidation plan | lifecycle, provenance, hash/tamper/version tests | calibrated physical soundness | Software profile yes | **blocked-by-calibration** |
| A15 | task representation is separate; exact T9A realizability is conditional | feedforward actor, certifier isolation, branched trainer | source, gradient-boundary, and accepted/fallback tests | exact truncated-Boltzmann realizability and training evidence | No T9A upgrade | **partial** |

#### Lemmas L0--L7

| Item | Current Mathematical Premise | Current Code Object | Current Evidence | Missing Content | Can Close This Round? | Status |
|---|---|---|---|---|---|---|
| L0 | Conditional grid/corridor update soundness under \(\Theta_s\) | calibration sensor contract and `RollingLocalGeometry` | split/semantic/domain/version plus geometry/provenance tests | real calibrated \(\Theta_s\) and coverage evidence | Software implication yes | **blocked-by-calibration** |
| L1 | Outward successor outer enclosure | `Interval`, `Interval3`, envelope builder | analytical and containment tests | calibrated physical bounds; nonlinear verifier if dynamics change | Software affine implication yes | **blocked-by-calibration** |
| L2 | Free stopping tube excludes obstacles and unknown | full-cell `box_is_verified_free` | unknown/partial/dilation tests | L0 physical validity | Conditional yes | **partial** |
| L3 | Every action in complete set preserves one-step collision safety | `verify_complete` plus L1 | full-zonotope interval tests | L0/L1 physical validity | Conditional yes | **partial** |
| L4 | Complete-cell \(\kappa\) successor remains in corridor | `RecoveryCellCertificate`, cell verifier | all-cell enclosure/hash/version tests | calibrated L0/L1 | Software conditional yes | **blocked-by-calibration** |
| L5 | Current linked proof objects authorize \(\kappa\) | `certified_action` requires recovery+energy hashes | actor-independent and stale-version tests | physical L4/L6/T5 premises | Boolean gap closed; physical no | **blocked-by-calibration** |
| L6 | Complete successor reaches lower level | cell verifier successor-ID inclusion | descent-failure rejection | real corridor verification | Software conditional yes | **blocked-by-calibration** |
| L7 | E3 preserves \(e\ge\overline R+e_G+m\) | energy certificate plus zonotope/recovery checks | insufficient-reserve and E3 tests | calibrated A2 and reserve | Software conditional yes | **blocked-by-calibration** |

#### Theorems T0--T14A

| Item | Current Mathematical Premise | Current Code Object | Current Evidence | Missing Content | Can Close This Round? | Status |
|---|---|---|---|---|---|---|
| T0 | Atomic accepted branch lies in \(C_{\rm run}\); all certified outcomes lie in \(\mathcal A_{\rm cert}\) | M1 map, L5a verifier, runtime/watchdog | membership, timeout, stale, blocking, one-shot tests | physical L5/L5a and hard RT publication | Mechanism conditional yes | **partial** |
| T1 | Certified initial state remains jointly certified | no single object; L0--L7 + T0 chain | component tests only | calibrated end-to-end premises | No | **unresolved** |
| T2 | Every certified successor retains \(\kappa\) | proof-carrying corridor and `certified_action` | actor-independent recovery tests | physical L5 validity | Conditional software yes | **blocked-by-calibration** |
| T3 | Corridor-wide level certificates imply finite arrival | recovery certificate DAG in proof manifest | all-cell closure and lower-successor rejection tests | real calibrated certificate chain | Conditional finite proof yes | **blocked-by-calibration** |
| T4a | One-step DAG has unique finite backward solution | `RecoveryEnergySolver` and manifest-linked energy DAG | order/E3/tamper/whole-corridor tests | physical cost bounds | Algorithm closed | **blocked-by-calibration** |
| T4b | Stopped \(M\)-step operator has well-founded descent | no implementation | theorem proof only | stopped-block continuous verifier | No | **theory-only** |
| T5 | E3 certificate upper-bounds robust transit energy | energy contract, solver, residual verifier, manifest | underestimation, residual, reduced-value, and hash tests | calibrated A2 and real cells | Software conditional yes | **blocked-by-calibration** |
| T6 | Recovery reaches \(\mathcal G\) collision-free with energy | synthetic composition T3--T5 in closure pipeline | complete synthetic manifest only | all calibrated L0--L7 and real terminal continuation | No physical closure | **unresolved** |
| T7 | Three-mode guard/watchdog preserves T1 and recovery invokes T6 | deterministic harness, watchdog, WCET and adapters | full software failure-injection matrix | deployed mode/hysteresis, RTOS WCET, atomic I/O | No deployment closure | **blocked-by-deployment-evidence** |
| T8A | Frozen constrained soft operator has unique fixed point | theorem only | proof audit | no runtime need; formal MDP instantiation absent | Not this phase | **theory-only** |
| T9A | Exact optimizer improves within same feasible class | theorem only | proof audit | Gaussian realizability/training | No | **theory-only** |
| T10A | General pushforward theorem | theorem only | proof audit | extension implementation | Not selected | **theory-only** |
| T11A | Exact truncation baseline | none | none | baseline implementation | Not primary | **control/baseline-only** |
| T12A | Accepted affine-tanh branch has exact density/gradient | `actor.py`, accepted branch in `trainer.py` | four CPU-Torch formula/gradient/epoch/branch tests executed | does not imply T9A full-class realizability | Selected-map formula closed | **implemented** |
| T13A | Projection baseline membership under verified target | none | none | baseline implementation | Not primary | **control/baseline-only** |
| T14A | State-level accepted branch plus deterministic \(\kappa\) atom | runtime/replay/watchdog/trainer | branch, fallback-atom exclusion, failure-injection, and replay tests | learned continuous \(\beta_\theta\) intentionally absent; no unified mixed entropy objective | Current 0/1 implementation yes | **partial** |

No continuous-domain or physical row is upgraded by a loss curve, random rollout, finite candidate set, successful optimizer trace, or unit test. Tests establish software properties of the finite reference profile only.

## Physical Contract Calibration, Single-Corridor Closure, and Trainer Integration

### Round baseline audit

The baseline was captured before this round's edits. It used Python 3.12.3 on Linux/WSL2 (`6.6.87.2-microsoft-standard-WSL2`), x86-64, Intel i5-13400F, 16 logical CPUs, 8 cores, and 20 MiB L3. Torch was absent in that interpreter.

| Item | Current Code | Current Evidence | Current Status | This Round's Target |
|---|---|---|---|---|
| Explicit geometry and interval core | `geometry.py`, `interval.py`, `envelope.py` | 42-test baseline: 40 passed, 2 Torch tests skipped | blocked-by-calibration | Bind all bounds to evidence-bearing contracts |
| Recovery and energy proof objects | `recovery.py`, `energy.py`, `certificates.py` | complete-cell and E3 unit fixtures | blocked-by-calibration | Close one fixed corridor into one linked manifest |
| Generator runtime | `zonotope.py`, `runtime.py`, `watchdog.py` | membership, version, timeout, one-shot tests | partial | Add evidence hashes, WCET profiling, and closed-loop audit |
| T12A actor path | `actor.py` | two tests skipped because Torch was absent | partial | Execute Torch formula/gradient tests and integrate trainer epoch semantics |
| Calibration acquisition | no unified package | no real evidence | unresolved | Add schemas, validation, reports, loaders, and synthetic execution fixtures |
| HIL/deployment | simulated watchdog only | no RTOS, bus, or HIL evidence | blocked-by-deployment-evidence | Add adapters and benchmark/failure interfaces without upgrading physical claims |

Baseline command and result:

```text
python3 -m unittest discover -v -s tests -p 'test_*.py'
Ran 42 tests in 0.720s
OK (skipped=2)
```

### Calibration package and evidence semantics

The independent `calibration/` package has the following immutable/versioned objects.

| Artifact | Required contents | Strict use | Current evidence status |
|---|---|---|---|
| `EvidenceMetadata` | evidence ID; data interval; device/firmware; method; operating domain; confidence semantics; source kind; expiry; recalibration rule; digest | common provenance and applicability gate | software-verified |
| `SensorCalibrationContract` | \(\epsilon_p,\epsilon_\theta,\epsilon_r,\beta,\epsilon_t,r_{body},\epsilon_{grid},d_{max},v_{max},\tau_{age},n_{free},v_{cal}\) | constructs `SensorBounds`; version change clears unsupported FREE evidence | blocked-by-calibration |
| `TrackingCalibrationContract` | command, publication, measurement timestamps and coordinate-wise tracking radii | supplies \(\epsilon_a\) to `DynamicsBounds` | blocked-by-calibration |
| `DynamicsCalibrationContract` | initial-state radii, period interval, component latencies, residual/wind boxes, tracking version | constructs outward-rounded successor envelopes | blocked-by-calibration |
| `EnergyCalibrationContract` | avionics/hover, velocity/action, communications/compute, measurement and underestimation margins | robust one-step \(\bar c\), T4a solver, and E3 verifier | blocked-by-calibration |
| `TerminalCalibrationContract` | horizontal set, altitude set, velocity box, \(e_G\), evidenced continuation modes | defines `ChargeAdmissible`; transit energy remains separate from \(e_G\) | blocked-by-calibration |
| calibration reports | split counts, operating domain, per-channel maximum residual, selected bound, validation exceedance, semantics, validity, notes, hash | audit artifact; never substitutes for the contract premise | software-verified schema; physical data absent |

The package distinguishes four semantics:

1. an **empirical quantile** describes only the supplied finite sample;
2. a **pointwise confidence bound** controls one declared query and is insufficient for a corridor-wide theorem;
3. a **simultaneous confidence bound** declares a family and \(\delta\), but still requires justified sampling/exchangeability assumptions;
4. a **deterministic engineering bound** requires a separately sourced engineering argument and observed-domain validation.

`build_sensor_contract` rejects attempts to label empirical/probabilistic estimates as deterministic. `CalibrationRegistry` rejects reuse of one `(kind, version)` pair with different evidence hashes. Every contract checks its data split, domain, device/firmware, expiry, and exact content hash. Synthetic or replay evidence always reports `blocked-by-calibration`; `allow_synthetic=True` exists only for deterministic software fixtures.

Raw CSV interfaces distinguish commanded, published, and measured acceleration; align state, actuator, voltage/current, and energy timestamps; and retain train/calibration/validation labels. Outlier removal is restricted to documented instrument faults. Winsorized or silently discarded tail data cannot certify an outer bound.

### Calibration estimation and validation contracts

For scalar residual samples \(r_j\), a selected strict bound \(b\) is accepted by software only if the independent validation report records every exceedance \(\mathbf 1\{r_j>b\}\), the operating point is in domain, and the declared confidence semantics match the estimation method. A zero observed exceedance is not automatically a deterministic bound.

For dynamics samples with measured actuator response \(a^{meas}\), the report computes

\[
r_p=\left|p^+-p-\Delta v-\tfrac12\Delta^2a^{meas}\right|,
\qquad
r_v=\left|v^+-v-\Delta a^{meas}\right|,
\]

coordinatewise and stratifies records by declared flight mode and payload. The strict runtime then propagates the initial intervals, full action interval, tracking radius, wind/residual boxes, and

\[
\Delta\in[
\Delta_0-\epsilon_\Delta,
\Delta_0+\epsilon_\Delta+	au_{sensor}+\tau_{compute}+\tau_{switch}
].
\]

For energy, validation detects any sample satisfying \(c_{meas}>\bar c(v,a)\). The runtime bound is

\[
\bar c(v,a)=c_{avionics}+c_{hover}
+\sum_i k^v_i\sup|v_i|
+\sum_i k^a_i\sup|a_i|
+c_{comm}+c_{compute}+\epsilon_{meas}+m_{under}.
\]

This is evaluated over the complete state/action cell. Expected energy fits are not accepted by T4a/T5. A changed energy hash invalidates recovery-energy and zonotope objects even if a human accidentally reuses the display version.

The first terminal profile permits a mode only when the contract includes a nonempty evidence identifier for that mode. Hover-only evidence may certify hover continuation. Docking and charging handoff remain unresolved until their own evidence exists; the package does not infer them from proximity to the station.

### Single-corridor closure pipeline

`SingleCorridorClosurePipeline` implements only A8-1/T4a. Given one fixed AABB chain and frozen κ, it executes the following fail-closed order:

1. validate all five calibration contracts, their hashes, domain, device/firmware, and expiry;
2. require the grid's active sensor version to match the sensor contract;
3. require runtime dynamics, energy, tracking, terminal, and κ versions to match;
4. create the complete ordered corridor; every cell must be FREE and every adjacent transfer overlap must pass;
5. verify all corridor state cells under the full interval action of κ;
6. reject the entire corridor if any actuator, stopping-tube, velocity, energy-floor, terminal, or one-step lower-level predicate fails;
7. install linked `RecoveryCellCertificate` objects only after all cells pass;
8. solve the T4a recursion in increasing level order and verify every E3 residual with outward rounding;
9. install `RecoveryEnergyCertificate` objects only after all residuals pass;
10. authorize κ from the installed linked objects;
11. construct and completely verify the diagonal full-rank zonotope;
12. emit a proof manifest only if every predecessor hash exists and the manifest hash verifies.

The failure output contains the first failed cell, predicate, interval enclosure, involved versions, required margin, and actual margin when that predicate supplies scalar margins. A failed cell is never skipped.

### Proof manifest format and invalidation

Every manifest entry records:

- object ID and object type;
- certificate epoch;
- sensor, dynamics, tracking, energy, terminal, geometry, corridor, and κ versions where applicable;
- creation time, expiry, and invalidation reason;
- predecessor proof hashes;
- its own proof hash and status.

The generated dependency DAG is

\[
\text{calibration contracts}
\to \text{proof-carrying grid}
\to \text{recovery-cell DAG}
\to \text{recovery-energy DAG}
\to \text{zonotope certificate}.
\]

There is no reverse edge from safe arrival to the energy solver and no edge from projection/mapping to set safety. `dependency_invalidation_plan` maps each changed version or hash to all affected successor, geometry, corridor, recovery, energy, and zonotope objects. `ReturnCorridor.invalidate_certificates` removes runtime authority after an applicable change.

The reproducible synthetic closure command is:

```bash
.venv/bin/python scripts/run_certificate_closure.py --output /tmp/synthetic-proof-manifest.json
```

It currently emits an 11-entry, 18-edge, hash-valid manifest and status `conditionally-verified-blocked-by-calibration`. This is a complete software fixture, not a flight corridor certificate.

### Measurability of the selected finite mechanism

For the selected profile, the strict input encoding consists of finitely many real interval endpoints, ternary cell labels, finite AABB records, integer versions, and finite proof identifiers. With the product Borel sigma-algebra, this is a standard Borel encoding. Fixed-iteration lexicographic bisection applies affine interval arithmetic, finite `min`/`max`, and threshold predicates, all Borel maps. Every tie uses the fixed axis order \(1,2,3\) and the fixed accept-low/reject-high rule. Therefore the implemented \(z\mapsto(c(z),G(z))\) is Borel measurable on its finite-profile generator-enabled domain, provided nonfinite inputs are excluded and each continuous-set predicate is the implemented Borel affine/AABB predicate.

This closes A10a only for the selected finite affine/AABB program. It does not establish continuity, a safeguarded MDP kernel, or measurability of future nonlinear branch-and-bound plugins. Positive volume is claimed only after the actual certificate records \(g_i\ge\sigma_G\), giving

\[
\lambda_A(C_{run}(z))=8|\det G(z)|=8g_1g_2g_3\ge8\sigma_G^3.
\]

No positive-volume claim is made on `NO_GENERATOR_SET` states.

### Zonotope deployment evidence

The constructor retains deterministic coordinate order and at most \(2+3I\) complete verifier calls. A successful object records actuator inclusion, \(\sigma_{min}\), condition upper bound, complete interval successor, stopping-tube containment, corridor predecessor inclusion, energy reserve, recovery hash, bound fingerprints, elapsed construction time, and deadline. `GeneratorRuntimeStatistics` accumulates enabled/no-set/fallback fractions, \(8|\det G|\), singular values, condition numbers, iteration/time data, and failure predicates. These software statistics are deployment diagnostics, not proof that real calibrated bounds hold.

### WCET and atomic execution evidence

`WCETContract` now separates

\[
T_{sensor},T_{update},T_{\kappa},T_{corridor},T_{energy},T_{set},T_{actor},T_{recheck},T_{publish},T_{margin},
\]

and requires their sum to be strictly below \(\Delta_{ctl}\) with a deployment evidence identifier. `WCETBenchmarkHarness` supports warm-up, optional CPU affinity, multiple runs, input-size sweeps, median, p99, maximum, and per-stage maxima. On Python/Linux it always reports `blocked-by-deployment-evidence` unless a separately qualified hard-real-time platform contract is supplied. The example profile command is:

```bash
.venv/bin/python scripts/benchmark_cert_runtime.py --runs 20 --warmup 3
```

The final desktop fixture check used five measured runs at each of two input sizes (10 stage-size samples). Its summed per-stage maximum was 0.014109400 s, status was `blocked-by-deployment-evidence`, and `deployment_qualified` was false. This is a profiling observation, not a WCET upper bound.

`SimulatedWatchdog` records that κ was staged before the worker, accepts only a complete immutable matching bundle, rejects late/stale/exceptional output, and uses a one-shot publisher. This is software-verified fail-default logic. It is not RTOS preemption, hard WCET, memory-model, or actuator-bus atomicity evidence.

### Generator-SAC trainer semantics

`CertificateEpoch` hashes the full certificate snapshot: geometry/corridor versions and digests plus all bound versions and evidence hashes. `GeneratorSACTrainer.begin_epoch` freezes this object. Every optimization batch must match it exactly; changed certificates require rebuilding replay interpretation and starting a new operator epoch.

The accepted branch implements

\[
u=m_\theta(o^{task})+s_\theta(o^{task})\odot\epsilon,
\quad \eta=\tanh u,
\quad a=c+G\eta,
\]

\[
\log q_G(a\mid z)
=\log\phi_\theta(u\mid o^{task})
-\sum_i\log(1-\tanh^2u_i)
-\log|\det G|,
\]

with the stable identity

\[
\log(1-\tanh^2u)=2(\log2-u-\operatorname{softplus}(-2u)).
\]

The actor update detaches \(c,G\). The critic always consumes the replayed \(a_{exec}\). Current acceptance is state-level 0/1 and independent of \(u\). Fallback transitions train the critic with the actual κ/emergency action but are excluded from the Generator density/entropy term because they are atoms of a different reference measure. No differentiable \(\beta_\theta\) is fabricated. T12A is therefore implemented for the accepted affine-tanh branch; T9A full feasible-policy-class optimality remains theory-only.

### Closed-loop and HIL interfaces

The deterministic harness performs version validation, geometry update, corridor closure, staged κ, recovery/energy authorization, complete zonotope construction, watchdog execution, one-shot publication, tracking-adapter readback, replay write, and audit logging. It injects missing/expired calibration, pose jump, stale/invalid LiDAR, corridor invalidation, energy/dynamics version changes, recovery hash tampering, E3 failure, no generator set, actor NaN/timeout, certifier block/exception, certificate mutation, and publish timeout. Every injected software failure is tested to avoid task publication and produce an explicit logged reason.

The HIL adapter boundary contains state, LiDAR, actuator sink, energy, tracking, timestamp, watchdog, and log interfaces with mock/replay implementations. A future HIL campaign must cover stale sensing, pose jumps, wind, tracking degradation, corridor invalidation, narrow passages, deadline overruns, energy underestimation, repeated κ takeover, and no-positive-volume states. Such results are empirical/deployment evidence only.

### Updated evidence gates

| Gate | Required evidence | Current status | Blocking item |
|---|---|---|---|
| Software invariant | deterministic tests, proof hashes, invalidation, fail-closed logic | implemented | no physical implication |
| Bound identification | real calibrated sensing, tracking, dynamics, energy, terminal contracts | blocked-by-calibration | no real datasets/evidence IDs supplied |
| Corridor certificate | one manifest using physically valid contracts and real corridor geometry | blocked-by-calibration | synthetic manifest only; physical bounds blocked |
| WCET | flight-platform stage maxima, scheduler and command-bus evidence | blocked-by-deployment-evidence | desktop Python profile is not hard WCET |
| Closed-loop integration | atomic real I/O plus measured tracking logs | blocked-by-deployment-evidence | mock/replay adapters only |
| HIL | adversarial HIL campaign and retained logs | unresolved | no HIL connection or runs |
| RL training | T12A tests, frozen epochs, `a_exec` critic/replay | implemented as minimal interface | no large-scale training requested; T9A realizability unclaimed |

### Current test evidence

Two interpreters are reported separately:

```text
# System Python without Torch
Ran 66 tests in 1.008s
OK (skipped=4)

# UV-managed .venv with Torch 2.7.1+cu128, after acceptance/smoke integration
Ran 100 tests in 131.893s
OK
```

Thus the authoritative `.venv` run has 100 passed, 0 skipped, and 0 failed. The Torch tests cover stable tanh/log-determinant density, finite-difference/autograd agreement, frozen \(c,G\), executed-action critic input, certificate-epoch rejection, fallback-atom exclusion, finite actor/critic updates, and mixed-epoch refusal. The acceptance tests additionally cover the ten-scenario fail-closed matrix, reward/certificate isolation, and absence of direct ground-truth map synchronization. Tests not run are real calibration acquisition, real fixed-corridor closure, RTOS/WCET qualification, actuator-bus atomicity, and HIL/flight validation.

Random interior/corner tests remain bug-finding diagnostics. Mathematical completeness for the selected affine propagator follows from interval extension of the complete action box; physical applicability still depends on calibrated containment of the real system.

## Remaining Gaps

The recurrent-state bottleneck is removed by design. The dominant unresolved bottlenecks are now:

1. calibrating and validating the physical error contracts that make rolling local geometry and corridor proof objects sound;
2. proving deployment-time nonemptiness and completing real-time construction/publication of useful full-rank three-generator inner sets;
3. establishing measurable/continuous selection properties and hardware-valid set inclusion, singular-value, conditioning, and numerical bounds for \(c(z),G(z)\);
4. validating certificate-epoch turnover, accepted/fallback occupancy, and objective consistency in actual training without inventing a sample-dependent differentiable acceptance law.

### Mathematical gaps

1. **Sufficiency of the explicit set state.** Removing a neural latent eliminates an unverifiable theorem variable, but the finite representation of \(\mathcal M^{\rm local}\) and \(\mathcal C^{\rm back}\) must still be sufficient for the claimed set-valued transition. If two physical histories produce the same explicit set state but different admissible unseen geometry inside a region labeled free, L0 and A14 fail.
2. **Energy-free field input.** Because \(E_\psi\) excludes remaining energy, T5 requires a residual bound uniform over all certified full-state fibers sharing \(\bar z^{\rm cert}\). This uniformity is assumed, not yet constructed.
3. **Regularity of the generator correspondence.** A10a assumes measurable \(c,G\). Real-time inner-zonotope optimization may have nonunique solutions and discontinuous optimizer selection. A measurable tie-breaking rule is required for the Bellman kernel; continuity or local Lipschitz regularity is additionally needed for stable numerical implementation, but not for T8A's contraction itself.
4. **Positive singular-value certificate.** The theorem assumes \(\sigma_{\min}(G(z))\ge\sigma_G\) uniformly. Establishing a useful global \(\sigma_G>0\) may fail near narrow passages even when a single certified action \(\kappa(z)\) exists. Recursive feasibility therefore does not imply a positive-volume task-action set.
5. **Acceptance-law consistency.** T14A is exact only after \(\beta_\theta(z)\) and \(w_\theta^{\rm acc}\) are defined by the actual construction, verifier, deadline, and numerical-failure event. If acceptance depends on \(u\), T12A's unconditional generator density is not the conditional accepted density.
6. **Changing correspondence.** No monotone policy-improvement theorem compares epochs with different \(c,G\), geometry maps, corridor cells, transition envelopes, or feasible policy classes.
7. **Undiscounted uniqueness outside the corridor.** Without verified descent or properness, the recovery-energy equation can have multiple or infinite solutions.
8. **Joint convergence.** Existing two-timescale theory does not justify simultaneous nonlinear model, field, critic, feedforward actor, generator-construction, and candidate-recovery updates.
9. **Route-B bias.** A zero local penalty does not remove downstream resolvent bias; only the complete discounted path penalty has the stated meaning.

### Verification gaps

The reference code now implements finite continuous-set predicates for the affine/AABB profile, but the following **physical premises** still require calibration or independent continuous-region verification rather than sampled rollouts:

- sound LiDAR-to-set updates for \(\mathcal F,\mathcal O,\mathcal U\), including pose and range uncertainty;
- proof that unknown or unresolved cells are never promoted to \(\mathcal F\) without sufficient evidence;
- containment of the full delay/reaction/braking tube, not merely the scalar range inequality \(r_{\rm sens}\ge d_{\rm stop}\);
- sound update and overlap of every cell in \(\mathcal C^{\rm back}\), including velocity, energy, and braking envelopes;
- corridor-wide verification of \(\kappa\)'s tracking, speed limit, braking, obstacle avoidance, and level descent;
- verified successor containment for all actions in the complete zonotope, not only vertices or sampled actions unless a separate convexity theorem makes those checks complete;
- actuator-bound inclusion and robust predecessor inclusion of \(c+G[-1,1]^3\);
- lower bounds on \(\sigma_{\min}(G)\), upper bounds on conditioning, and measurable selection of \(c,G\);
- validity of the implemented full-cell E3 residual under calibrated real energy bounds;
- candidate recovery-policy replacement on continuous corridor neighborhoods;
- simultaneous probabilistic coverage if deterministic geometry, dynamics, or energy bounds are unavailable.

Finite rollouts, ensemble spread, sampled generator corners, or successful optimization traces do not establish these premises by themselves.

### Runtime and implementation gaps

- The selected ternary-grid and AABB-chain representation must still meet the real control deadline at deployment resolution and corridor length; the prototype supplies complexity accounting but not aircraft WCET evidence.
- `SimulatedWatchdog` demonstrates independent default-κ logic, immutable snapshots, complete-bundle acceptance, and one-shot publication, but a real independent scheduling domain, RTOS priority/preemption evidence, and atomic actuator-bus publication remain required.
- The reference constructor now uses deterministic lexicographic diagonal bisection around \(\kappa\). Its safety verifier is specified, but its conservativeness, discontinuities, and task-performance loss relative to a verified maximum-volume constructor remain unresolved.
- If no full-rank verified zonotope exists, the runtime must fall back to \(\kappa\); it must not reduce \(\sigma_G\) silently or use a rank-deficient determinant.
- The actor implements \(\log(1-\tanh^2u)=2(\log2-u-\operatorname{softplus}(-2u))\) without undocumented clipping. All four actor/trainer tests passed in the UV-managed `.venv`; this validates the implementation formulas, not T9A realizability or convergence.
- Replay stores the immutable \(z^{\rm cert}\) snapshot, \(o^{\rm task},u,\eta,c,G,a_M,a_{\rm exec},\kappa\), acceptance, fallback reason, certificate version, and recovery/inclusion hashes.
- The critic is trained on \(a_{\rm exec}\). A nominal-\(u\) critic would define a different safeguarded MDP and is not the primary architecture.
- Current construction acceptance is state-level and independent of \(u\); no differentiable \(\beta_\theta\) is claimed. If a future mechanism makes acceptance depend on \(u\), it needs an explicit conditional-density correction or a declared biased surrogate.
- T11A exact truncation and T13A projection require separate experimental implementations and must not share the primary generator's entropy formula.
- SEditor remains an empirical proposal editor unless its complete output image is independently verified before the generator/fallback layer.

### Empirical evidence requirements

The evidence layers remain separate:

| Layer | Evidence | What it establishes |
|---|---|---|
| Training loss | Field regression, Bellman residual, generator-construction objective | Optimization progress only |
| Empirical rollout | Collision rate, recovery success, energy reserve, task return, takeovers | Sampled performance only |
| Continuous verification | Set-update soundness, successor containment, complete-zonotope inclusion, \(\kappa\) progress | Premises of the corridor theorem |
| Mathematical proof | Set containment, induction, stopped-block descent, change of variables, contraction | Consequences conditional on verified premises |

Experiments must report the previously listed dynamics, energy, collision, recovery, task, and generalization metrics plus:

- free/obstacle/unknown classification errors and unknown-to-free promotion errors;
- age, length, update frequency, and invalidation rate of \(\mathcal C^{\rm back}\);
- minimum directional sensing slack relative to the complete braking tube;
- zonotope volume \(8|\det G|\), \(\sigma_{\min}(G)\), condition number, and inclusion-verification time;
- fraction of certified states with no full-rank zonotope despite a valid \(\kappa\);
- tanh saturation, log-Jacobian magnitude, and gradient variance;
- acceptance probability calibration, conditional-on-acceptance density checks, timeout rate, and fallback frequency;
- exact-truncation and projection controls using their own correct entropy objectives;
- separate reporting inside the certified corridor, in known but uncertified space, and in unknown space.

None of these measurements alone establishes deterministic global safety.

## Pending Design Choices

The following architecture choices are fixed: no recurrent certificate state; dual task/certificate pathways; explicit local geometry and sparse return corridor; feedforward actor; full-rank three-generator Route-A realization; exact truncation and projection only as controls.

| Choice | Remaining alternatives and consequence | Current recommendation |
|---|---|---|
| Collision semantic label | Robust viability/HJ value, verified braking margin, or discrete-time barrier | Use braking-tube semantics for the first strict theorem unless an HJ/viability equivalence is independently verified |
| Uncertainty guarantee | Deterministic outer sets or simultaneous \(1-\delta\) sets | Maintain parallel theorem statements; never convert statistical coverage into deterministic safety |
| Explicit set representation extensions | The executable profile fixes a ternary grid and AABB corridor; polytope/zonotope hybrids remain possible ablations | Keep the fixed profile for the first implementation and compare alternatives only after soundness/WCET evidence exists |
| Generator construction extensions | The executable profile fixes lexicographic diagonal bisection; verified maximum-volume or task-weighted constructions remain alternatives | Treat larger constructors as performance extensions, not changes to L5a's certificate source |
| Acceptance extensions | Runtime fixes construction-level acceptance independent of \(u\); actor-dependent rejection is only a failure branch | Keep T12A on the accepted branch; use T14A only if a future learned acceptance probability is introduced |
| Recovery-progress extension | Runtime fixes one-step descent; bounded-\(M\)-step verification remains available theoretically | Keep T4a for the executable profile and implement T4b only with a stopped-block continuous verifier |
| Recovery-policy replacement | Preserve the old corridor, verify overlapping handover regions, or allow shrinkage with migration | Require verified overlap and migration; full preservation is safest but may block improvement |
| Two-timescale claim | Frozen-epoch modular theorems or a full joint convergence theorem | Keep frozen-epoch results unless the paper explicitly claims joint convergence |
| Other UAVs | Continue static single-agent geometry or introduce coupled dynamic obstacles | Continue excluding inter-agent collision in the first theory |
| Energy risk | Expected, nested/CVaR, or robust worst-case recovery cost | Use robust worst case for the strict theorem; expected/CVaR variants remain performance controls |

## Boundaries and Non-Claims

- The main conclusion remains a **corridor-conditional robust guarantee**, not safety in a globally unknown environment.
- Unknown space is non-certifiable. The theorem does not guarantee entry into or traversal through \(\mathcal U\).
- The explicit rolling geometry and sparse return corridor are not claimed to be a complete global SLAM map.
- Removing the recurrent state removes one verification obstacle; it does not prove that the explicit geometry update is sound or Markov-sufficient.
- The feedforward CNN/MLP task representation is not a certificate and is absent from physical theorem premises.
- The learned collision and energy fields are proposal functions, not certificates.
- The generator map proves candidate membership in its image; it does not prove \(C_{\rm run}\subseteq\mathcal A_{\rm cert}\).
- Verifying only sampled actions, vertices, or trajectories does not verify the complete zonotope without an applicable completeness theorem.
- A valid fallback action \(\kappa(z)\) proves nonemptiness of \(\mathcal A_{\rm cert}(z)\), not existence of a positive-volume full-rank zonotope.
- T12A requires exactly three independent generators. Its determinant, density, and entropy formulas do not apply to \(g>3\), noninjective, rank-deficient, or dimension-changing maps.
- The selected affine-tanh map is a diffeomorphism, not an isometry.
- Nominal Gaussian entropy is not the executed-action entropy; both the tanh correction and \(-\log|\det G|\) are required.
- T12A gives the exact gradient for frozen \(c,G\). Allowing task gradients to change \(c,G\) invalidates that theorem and the set certificate.
- A diagonal-Gaussian latent policy need not realize T9A's exact truncated-Boltzmann optimizer.
- T14A does not justify ignoring \(\nabla_\theta\beta_\theta\) or using the unconditional generator density when acceptance depends on \(u\).
- T11A exact truncation and T13A projection are baselines/controls, not the primary implementation.
- Projection verifies neither its target set nor the UAV predecessor relation.
- SEditor does not provide zero-violation, recursive-feasibility, or safe-return guarantees.
- Standard monotone soft policy improvement is not claimed after changing \(c,G,C,P,r\), the certificate version, or the realizable actor class.
- Finite rollouts, ensemble spread, and empirical map accuracy do not supply deterministic global bounds.
- Strict safety outside the continuously verified corridor is not claimed.

## Open Risks

The largest risk is calibrating and maintaining a sound rolling partition of free, obstacle, and unknown space while preserving a sparse return corridor under sensing, pose, timing, and tracking uncertainty. The second is constructing and atomically publishing, before every deployment deadline, a useful full-rank inner zonotope whose complete successor image is certified. Narrow passages may leave \(\kappa\) valid while forcing every positive-volume task set to collapse, causing sustained recovery fallback. The third is future objective mismatch if acceptance is changed from the current state-level event to sample-dependent editing while training still uses T12A's unconditional density. These risks limit performance and theorem applicability but do not justify expanding the guarantee beyond the verified corridor.

## Finalized Architecture Summary

1. The certificate state is \(z^{\rm cert}=(p,v,e,p_{\mathcal G},\mathcal M^{\rm local},\mathcal C^{\rm back},\xi)\); no recurrent latent appears in strict theorems.
2. A feedforward CNN/MLP task path proposes \(u\), while an explicit geometric path constructs and verifies \(c,G,\kappa\).
3. Unknown space is excluded from certification; the braking tube and update latency are contained in verified free geometry.
4. On the generator-enabled domain \(\mathcal Z_G\), the primary candidate is \(a_M=c(z)+G(z)\tanh u\), with \(G\in\mathbb R^{3\times3}\), \(\sigma_{\min}(G)\ge\sigma_G\), and complete-zonotope inclusion in \(\mathcal A_{\rm cert}\). Outside \(\mathcal Z_G\), task execution is disabled and \(\kappa\) is used.
5. T12A supplies the accepted-branch density, entropy, and SAC gradient. T14A supplies the fallback mixture. T11A and T13A are controls; T10A covers nonbijective extensions.
6. Every construction, verification, rank, numerical, or deadline failure executes the independently certified \(\kappa(z)\).
7. Certificate construction, runtime enforcement, and RL optimization remain separate proof chains.
8. The strongest claim remains a corridor-conditional robust guarantee.
9. The executable profile uses a proof-carrying ternary local grid, an overlapping AABB return chain, affine interval successor propagation, and deterministic diagonal-zonotope enlargement around \(\kappa\).
10. The prototype implements proof-object generation, outward-rounded affine envelopes, complete-cell one-step recovery verification, T4a/E3 energy certificates, full-zonotope verification, and watchdog logic. Physical closure still requires real sensor/dynamics/energy calibration, platform-qualified numerical behavior, deployed corridor certificates, and hardware/RTOS WCET plus atomic-I/O evidence.

## Single-UAV Certification Experiment Environment

The executable architecture is instantiated in `envs/certified_uav/` without changing the legacy `envs/UAVEnergyDelivery.py` import path. The new plant accepts only \(a_{exec}\), uses the shared `integrate_double_integrator` function, rejects actions outside the actuator box, performs swept body collision, subtracts a separate realized simulation-energy cost, emits a versioned 32-ray `LidarPacket`, and never repairs collisions, respawns depleted vehicles, or assigns tasks. Task reward/observation and certificate/runtime logic are separate wrappers.

The deterministic `open_corridor` fixture closes a synthetic four-level T4a corridor manifest and enables a full-rank diagonal Generator. The `narrow_corridor`, `invalid_corridor`, `insufficient_energy`, actor-nonfinite, stale-version, and watchdog-deadline cases all fail closed. Synthetic historical LiDAR packets are used only to exercise rolling proof-carrying geometry; the certificate path does not read the plant obstacle list. The environment emits `CertificateEpoch`, exact action traces, executed-action replay, and synthetic calibration records. Full design and non-claims are documented in `docs/SINGLE_UAV_CERTIFICATION_ENV.md`.

This closes a software experiment fixture, not physical premise closure. All generated calibration evidence remains synthetic, complete-cycle desktop timing remains `blocked-by-deployment-evidence`, and the open fixture's deliberately simple vertical braking hierarchy is not claimed to be a calibrated return flight. No real flight safety or global unknown-environment guarantee follows.
## 2026-08-07 Single-UAV acceptance addendum

The implementation audit preserves the three proof chains. Certificate construction is represented
by the proof-carrying grid/corridor/energy/zonotope objects; runtime enforcement is represented by
the immutable bundle, staged recovery command, version recheck, and one-shot publisher; RL
optimization is restricted to a frozen `CertificateEpoch`. None of the three is used as a proof of
the other two.

The accepted Generator branch implements
\[
 a=c+G\tanh u,\qquad
 \log q_G=\log\phi_\theta(u\mid o)-\sum_i\log(1-\tanh^2u_i)-\log|\det G|.
\]
`c` and `G` are detached. Fallback records are atoms under the hybrid execution law: they update
the critic through the actual `a_exec` but do not enter this continuous-density entropy term.
Mixed certificate epochs are rejected rather than treated as one unchanged Bellman operator.

Current evidence labels are: action/runtime invariants **software-verified**; fixed scenario and
stress behavior **synthetic-validated**; finite smoke-training updates **empirical-training-only**;
physical bounds **blocked-by-calibration**; hard timing and atomic actuator I/O
**blocked-by-deployment-evidence**. The main theorem therefore remains a corridor-conditional
robust guarantee under independently valid bounds. Zero synthetic collisions or finite training
losses do not upgrade T1, T6, or T7 to real-flight claims.

## 2026-08-07 Multi-Step Generator-SAC implementation addendum

The single-step smoke optimizer remains only a formula/data-flow control. The formal implementation
in `cert_runtime/generator_sac.py` instantiates the frozen-correspondence Route-A operator over
multi-step trajectories with twin target critics. For transition (t), the critic input is the
physically executed (a_t^{\rm exec}), never the nominal latent or rejected candidate. The target is

\[
y_t=r_t+\gamma(1-d_t)\begin{cases}
\min_i Q_{\bar\psi_i}(o_{t+1},c_{t+1}+G_{t+1}\tanh u_{t+1})
-\alpha\log q_G(a_{t+1}\mid z_{t+1}),&z_{t+1}\in\mathcal Z_G,\\
\min_i Q_{\bar\psi_i}(o_{t+1},\kappa(z_{t+1})),&z_{t+1}\notin\mathcal Z_G.
\end{cases}
\]

Here (d_t) is the physical termination indicator. Time-limit truncation bootstraps by default and
is stored separately. The first branch uses next-state (c_{t+1},G_{t+1}), including the stable tanh
Jacobian and (-\log|\det G_{t+1}|). The second branch is an atomic fallback and has no Generator
density. Targets are detached, target critics are updated by Polyak averaging, and actor/temperature
updates are restricted to accepted Generator rows with detached (c,G).

This implements an engineering safeguarded-hybrid Bellman target. It does not prove A10c's global
safeguarded-MDP kernel, deep-network convergence, or monotone improvement when certificate epochs
change. Replay is grouped by epoch (or may be rejected/cleared under configured policies), so a
batch never silently identifies incompatible state-dependent action correspondences.

The training task adds explicit OUTBOUND and RETURN phases. These variables affect observations and
reward but are not certificate sources. The deterministic synthetic mission fixture demonstrates
that a mission can require many consecutive transitions and can complete task then return. Physical
theorems remain conditional on independently valid calibration, corridor, κ, energy, numerical, and
deadline premises.

The former 2,000-step result exposed that the fast synthetic waypoint κ was not a corridor-wide
L4/L6 certificate. That defect was closed before further comparison: Shield-SAC and Generator-SAC
now share the same finite, hash-linked, strict-descent κ manifest with complete synthetic
state-tube, geometry, energy, and terminal checks. The mandatory mission gate passes all four
scenarios. In the repaired 48-run reduced matrix and the subsequent 80-run 10k matrix, both
certified methods recorded zero sampled collisions, zero uncertified task publications, and zero
fallbacks with invalid κ. These observations validate the synthetic implementation chain only and
do not upgrade T1, T3, T6, or T7 without calibrated physical bounds and deployment evidence.

## 2026-08-07 Mission-certificate closure refinement

The mission fixture now distinguishes a one-step task-set test from a corridor-wide recovery
certificate.  For every declared recovery cell (B_\ell), the proof manifest stores a complete
ellipsoidal/interval successor enclosure, actuator and velocity bounds, verified-free swept-tube
containment, a lower-level dependency hash, a robust one-step energy upper bound, and an E3
residual.  The executable profile implements only strict one-step descent:

\[
  \operatorname{Post}(B_\ell,\kappa)\subseteq B_{\ell-1},
  \qquad
  \bar R(B_\ell)\ge \bar c(B_\ell,\kappa)+\bar R(B_{\ell-1}).
\]

The finite FREE geometry is a union of AABBs.  Complete swept-rectangle inclusion in that union is
decided by the finite partition induced by every AABB boundary; one midpoint per partition cell is
complete because membership is constant on each open cell.  These checks remain synthetic software
evidence because their physical error bounds are not calibrated.

**Task-aware-center proposition (performance only).**  Let (a_{\rm pref}(z)) be any measurable
task proposal and let a deterministic constructor choose (c=c(z,a_{\rm pref})) and full-rank
(G).  If an independent complete-set verifier establishes

\[
  c(z,a_{\rm pref})+G(z)[-1,1]^3\subseteq\mathcal A_{\rm cert}(z),
\]

then every affine-tanh candidate belongs to \(\mathcal A_{\rm cert}(z)\), irrespective of how the
proposal was produced.  The proposal changes performance but is not a certificate source.  During
the SAC update (c,G) remain frozen/detached, so T12A's density is unchanged.  This proposition
does not authorize gradients through the verifier or remove the need to certify \(\kappa\)
independently.

The final post-closure software suite ran 119 tests in 160.314 seconds with 119 passes, zero failures,
and zero skips. This test result supports code-path and synthetic proof-object claims only; it does
not alter the blocked-by-calibration or blocked-by-deployment-evidence statuses in the ledger.

The three-seed center diagnosis isolates the earlier zero-task-success mechanism. With a verified
braking center, the deterministic trajectory remains 0.8 m from the task for 400 steps. With the
verified task-oriented center, all three fixture seeds complete task and certified return in 226
steps. The 10k matrix then obtains task/return success 1.0 in mission-open and mission-obstacle,
while mission-narrow and mission-energy-tight intentionally return before task completion. This is
a performance result about center selection; complete-set verification remains the certificate.

`NO_GENERATOR_SET` is interpreted operationally: the configured deterministic constructor cannot
produce a verified positive-volume full-rank set at that state. It does not prove emptiness of the
entire continuous certified-action domain. In mission-narrow, the state-uncertainty box intersects
a conservative task-authority exclusion while the independent κ certificate remains valid, which
implements the distinction that this constructor has no runtime task set while
\(\kappa(z)\in\mathcal A_{\rm cert}(z)\).

## 2026-08-07 RL-Contribution and Generalization Refinement

For the selected implementation,

\[
 a^{\rm exec}=c(z,a_{\rm pref})+G(z)\eta_\theta(o),\qquad \eta_\theta=\tanh u_\theta.
\]

This is interpreted as a **verified task reference plus a learned residual**, not as evidence that
the residual improves performance. `center_only` fixes \(\eta=0\), and `random_generator` uses an
untrained Gaussian latent. Both share the same independently verified \(c,G,\kappa\) and manifest
as Generator-SAC. Their ablation is therefore required before attributing efficiency to SAC.

The task proposal may affect \(c\), but the safety premise remains the independently verified
inclusion
\[
 c(z,a_{\rm pref})+G(z)[-1,1]^3\subseteq\mathcal A_{\rm cert}(z).
\]
The proposal is not a certificate, and task gradients do not pass through \(c\), \(G\), or the
verifier. This is a refinement of T0/T12A, not a new convergence theorem.

Intervention is partitioned into task intervention during OUTBOUND, planned recovery handoff
during RETURN, and failure fallback caused by certificate, version, numerical, or deadline failure.
These categories are not interchangeable. Scenario-family replay stores scenario ID, family,
scenario hash, and manifest hash; batches remain grouped by the frozen manifest epoch. No monotone
improvement claim is made across manifests. Out-of-contract disturbance invalidates the synthetic
gate rather than being counted as certified robustness.

The first RL-contribution ablation supports only a limited performance claim. Across five seeds and
20 evaluation episodes per seed, Center-Only, Random-in-Generator, and Generator-SAC all obtain
task/return success 1.0 with zero sampled collision in mission-open and mission-obstacle.
Generator-SAC matches Center-Only's 226-step open mission and improves the obstacle mean by only
0.15 step, approximately 0.00040 m path length, and 0.00164 synthetic energy units. The open return
is slightly lower because the stochastic residual adds cost without improving completion. Thus the
present handcrafted fixtures are dominated by the task-aware center. The supported contribution is
certified task-aware action-set construction, reduced OUTBOUND intervention relative to Shield, and
a semantically correct residual optimizer; material RL efficiency gains remain unestablished.

The held-out scenario-family experiment sharpens this non-claim. With one frozen 10k checkpoint
seed and 20 episodes on each of 20 independently certified held-out scenarios, Generator-SAC and
Center-Only have identical task-success patterns: 1.0 in open and obstacle, 0 in narrow, and 0.20
in energy-tight families. Return success is 1.0 throughout. Thus the present evidence does not show
that the learned residual improves generalization beyond the explicit center. It does show that
the same verified support and independently certified recovery authority remain usable after
bounded synthetic changes of start, task, obstacle realization, tracking, sensing, and energy.

This evidence supports the following attribution boundary:

1. corridor-wide certification and complete-set verification support the conditional membership
   and recovery claims;
2. task-aware center construction supplies nearly all demonstrated task competence;
3. the SAC residual has the correct hybrid Bellman and density semantics, but its demonstrated
   efficiency gain is negligible;
4. sampled held-out success and zero sampled collision are empirical observations, not replacements
   for calibrated envelopes or deployment timing premises.

Accordingly, 100k--200k training is not yet the next justified step. A harder certified family in
which Center-Only is feasible but suboptimal is required before spending a larger budget to test
whether residual learning adds value. The 50k multi-scenario interface remains available, grouped
by immutable manifest epoch, but has not been used to manufacture a positive RL claim.

## Persistent recoverability formulation (implementation-stage addendum)

Let the environment generate goal-reaching tasks \(\tau_1,\tau_2,\ldots\) for one UAV on a finite
certified goal network. The policy neither selects nor schedules tasks. Completing \(\tau_i\)
assigns \(\tau_{i+1}\), excluding the charging station from normal goals. One continuous policy
emits only \(u_t\in\mathbb R^3\), with

\[
\eta_t=\tanh u_t,
\qquad
a_t=c(z_t)+G(z_t)\eta_t.
\]

The persistent default center is safety-neutral: it supports a nondegenerate verified action set but
does not encode the task-goal direction, station direction, or a charging decision. Task flight,
voluntary station approach, charger dwell, departure, and task resumption must arise from this same
continuous policy. The categorical energy-management SMDP is retained only as an ablation and is
not part of the main theorem chain.

The recovery-energy field retains its undiscounted robust first-passage definition under frozen
\(\kappa\). Define the certified recoverable set

\[
\mathcal R
=
\left\{z:
\begin{array}{l}
\text{a valid certified recovery chain for }\kappa\text{ exists},\\
\text{the linked collision/geometry recovery certificate is valid},\\
e-E^\kappa(z)-e_G-m_e\ge 0
\end{array}
\right\}.
\]

Membership in \(\mathcal R\) means that \(\kappa\) is available as a certified backup; it does not
mean that \(\kappa\) currently controls the vehicle. Using the uncertainty-aware successor envelope,
define

\[
\mathcal A_{\rm rec}(z)
=
\{a:\operatorname{Post}(z,a)\subseteq\mathcal R\}.
\]

Thus the candidate-action verifier jointly checks actuator and velocity bounds, swept FREE geometry,
tracking/dynamics uncertainty, and, for every robust successor,

\[
e^+_{\rm lower}
\ge
E^\kappa(z^+)_{\rm upper}+e_G+m_e.
\]

Checking only the current reserve is insufficient. The complete Generator must satisfy

\[
C_{\rm run}(z)=c(z)+G(z)[-1,1]^3\subseteq\mathcal A_{\rm rec}(z).
\]

**T_REC1 (one-step recoverability preservation).** Assume \(z_t\in\mathcal R\) and runtime publishes
\(a_t\in C_{\rm run}(z_t)\), where the complete-set certificate proves
\(C_{\rm run}(z_t)\subseteq\mathcal A_{\rm rec}(z_t)\). Then every state represented by the certified
successor envelope belongs to \(\mathcal R\); in particular, the realized successor satisfies
\(z_{t+1}\in\mathcal R\) whenever the physical bounds are sound.

*Proof sketch.* Set containment gives
\(a_t\in\mathcal A_{\rm rec}(z_t)\). By definition of \(\mathcal A_{\rm rec}\),
\(\operatorname{Post}(z_t,a_t)\subseteq\mathcal R\). Soundness of the successor envelope places the
realized successor in that envelope. This argument uses certificate construction and set
containment, not RL contraction.

**T_REC2 (recursive recoverability under learned actions).** If \(z_0\in\mathcal R\) and every normal
learned action is published from a newly verified \(C_{\rm run}(z_t)\subseteq\mathcal A_{\rm rec}(z_t)\),
then \(z_t\in\mathcal R\) for every normal-policy step.

*Proof sketch.* Apply T_REC1 at \(t=0\); use its conclusion as the induction premise at the next
step. The theorem preserves the existence of a certified recovery option. It does not claim that
the learned policy itself reaches the terminal.

Normal RL authority requires an interior energy margin. At the configured switching boundary, or
after `NO_GENERATOR_SET`, evidence/version failure, deadline failure, or atomic publication failure,
authority switches to frozen \(\kappa\). Since switching occurs from \(\mathcal R\), the existing
strict corridor descent, E3 recursion, and finite-time terminal-arrival theorem apply. An invalid
\(\kappa\) certificate instead causes fail-closed termination.

Voluntary station approach while the margin remains interior is not recovery takeover: the
Generator policy retains physical authority. Remaining inside the charge-admissible set applies
\(e_{t+1}=\min\{e_{\max},e_t+r_c\Delta t\}\). While departure is closed, construction uses
\[
C_{\rm charge}(z)\subseteq\mathcal A_{\rm rec}(z)\cap
\{a:\operatorname{Post}(z,a)\subseteq\mathcal G_{\rm charge}\}.
\]
Thus an accepted normal policy sample cannot request an executable departure and is not post-hoc
aliased to another action. When the version-matched departure-energy gate opens, normal
\(C_{\rm run}\subseteq\mathcal A_{\rm rec}\) is restored. Station hold is retained only for failure
of the constrained support or runtime machinery. Future charging never reduces current return
energy.

### Persistent Bellman/runtime authority consistency

The pure execution classifier has four outcomes: `RL_GENERATOR`, `KAPPA_BACKUP`,
`CHARGER_CONSTRAINED`, and `FAIL_CLOSED`. It evaluates the persistent manifest, recovery and
recoverable-set certificates, complete \(\mathcal A_{\rm rec}\) inclusion, policy-authority gate,
energy switching margin, charging state, and departure gate. Runtime stores that result in the
next-state replay context; the Bellman target does not reconstruct authority from only
`next_generator_available`.

If the recorded next authority is `RL_GENERATOR`, or `CHARGER_CONSTRAINED` with a verified
full-rank \(C_{\rm charge}\), the target samples from that next-state affine-tanh support and includes
its continuous entropy. If it is `KAPPA_BACKUP`, the target action is exactly
\(\kappa(z_{t+1})\) and has no Generator entropy. An exceptional atomic charger hold likewise has
no Generator density, and `FAIL_CLOSED` has no bootstrap continuation. This aligns the engineering
Bellman branch with actual publish authority; it does not add a convergence theorem and does not
enter T_REC1/T_REC2's safety proof.

`PERSISTENT_SAFETY_GATE` binds task, recovery, and departure routes using typed prerequisites.
Shared dynamics, tracking, energy, terminal, recoverable-set/action-rule, and runtime versions must
match globally. Edge geometry, corridor, mission-manifest, and \(\kappa\) identities may differ, but
their hashes are bound per edge and included in the aggregate manifest. A `RECOVERY_EDGE` proves the
complete \(\kappa\) chain, strict descent, geometry, E3, and terminal linkage; it does not assume a
positive-volume Generator. `TASK_EDGE` and `DEPARTURE_EDGE` additionally bind their normal-authority
successor support.

`POLICY_AUTHORITY_GATE` checks only task/departure/charging states where normal RL authority is
claimed for a three-dimensional neutral-center, full-rank nondegenerate \(G\), task- and
station-directed residual authority where meaningful, and complete-set recoverability.
`POLICY_AUTHORITY_COVERAGE` separately reports the fraction of eligible roots with such support.
Thus \(z\in\mathcal R\) does not imply that a Generator exists: `NO_GENERATOR_SET` with a valid
\(\kappa\) invokes backup and does not violate T_REC1/T_REC2. These are synthetic software contracts.
Physical premises remain blocked-by-calibration and deployment timing remains
blocked-by-deployment-evidence.

The corrected synthetic validator was executed. `persistent_open` and `persistent_energy_tight`
pass `PERSISTENT_SAFETY_GATE` and `POLICY_AUTHORITY_GATE`; each has 1353/1353 RL-authority roots and
36984/36984 kappa-only cells valid. `persistent_obstacle` remains blocked: the first witnesses on
`recover_C_S`, `task_C_B`, `task_C_D`, and `task_D_C` have `minimum_geometry_slack=-1.0`, while hash,
E3, velocity, and descent checks pass. Across those edges, respectively 906, 591, 1017, and 809
cells fail complete swept-geometry containment. This is synthetic certificate infeasibility and was
not altered to force PASS. The implementation regression is 187 passed, zero skipped, zero failed
in 210.896 seconds. Acceptance and training were not executed.

## Task-independent random persistent extension

Let `x` denote physical/certificate state and `g` the externally assigned task goal. The learned
policy may depend on both, while the certified correspondence does not:

\[
\pi_\theta(a\mid x,g),\qquad
\mathcal A_{\rm safe}(x)=\mathcal A_{\rm act}(x)\cap
\mathcal A_{\rm col}(x)\cap\mathcal A_{\rm rec}(x),\qquad
C_{\rm run}(x)\subseteq\mathcal A_{\rm safe}(x).
\]

The task-independent recovery atlas is constructed only from certified FREE geometry, terminal
geometry, dynamics/tracking/energy envelopes, runtime bounds, and frozen `kappa`. Task goals,
task rewards, task edges, task waypoints, and task-route successors are excluded from its manifest
and support constructor. The legacy fixed graph remains a regression fixture, not a premise of the
random-goal method.

**T_RAND1 (random certified initialization).** If `supp(mu_0) subset R`, then a sample from `mu_0`
satisfies the initialization premise of T_REC1 and T_REC2. This does not certify points outside the
atlas.

**T_RAND2 (goal-independent recursive recoverability).** For any admissible goal process and any
goal-conditioned policy, if each normal action belongs to a freshly verified
`C_run(x_t) subset A_rec(x_t)`, then `x_t in R` for all normal RL steps. The proof is T_REC2
induction because the goal is absent from the certified successor premise. This guarantees
recoverability, not task completion or learnability.

**T_RAND3 (goal-independent support contract).** For fixed `x` and evidence versions,
`C_run(x,g_1)=C_run(x,g_2)=C_run(x)`. The software gate checks equality of `E^kappa`, recoverable
membership, kappa proof identity, `c`, `G`, action bounds, and atlas identity. Goal-conditioned
actor latents may differ.

Reported atlas coverage is a synthetic discretized capability metric, not global free-space
certification.

## Persistent authority lifecycle closure

Recoverability and normal-policy viability are distinct.  The certified recovery domain remains
`R`; the task-independent normal-authority domain is the greatest finite atlas fixed point
`R_RL subset R` whose cells have a nondegenerate complete Generator and a verified continuation
successor.  Recovery cells outside that fixed point remain valid kappa-only states.  The
`safety_neutral` center is computed only from the active atlas reference and physical state; it is
independent of the sampled goal, reward, task edge, and task waypoint.

Define `A_cont(x) = {a: Post(x,a) subset R_RL union G_charge}`.  Normal support must satisfy both
`C_run(x) subset A_safe(x)` and `C_run(x) subset A_cont(x)`.  The implementation binds the selected
continuation cell and complete successor certificate into the action context.  `NO_GENERATOR_SET`
in a kappa-only state is still a valid backup condition rather than a recovery-certificate failure.

**T_AUTH1 (normal-authority viability).** If `x_t in R_RL` and the runtime publishes from a freshly
verified `C_run subset A_safe intersect A_cont`, then every robust successor is in `R_RL` or the
certified charging set.  This is an authority-viability statement, not task convergence.

**T_AUTH2 (zero-step terminal recovery).** If the complete certificate-state uncertainty set is
contained in `G_charge`, the terminal recovery certificate has level zero, no successor, and
`E^kappa=0`.  Terminal reserve and version/hash requirements remain active; charging is not a bare
boolean bypass. Zero recovery energy does not mean zero physical hold action: the certificate also
binds a local position/velocity hold law and requires its complete successor envelope to remain in
`G_charge`. If that invariance check fails, the terminal certificate is not valid for that state.

**T_AUTH3 (safe lifecycle closure).** With valid evidence, normal RL authority may switch to kappa,
kappa reaches the terminal, the zero-step certificate closes recovery, charger-constrained support
or certified hold maintains the station set, and a verified departure returns to `R_RL`.  No
task-specific route enters this lifecycle contract.

### Task-control authority is separate from safety

The random-persistent atlas uses zero-velocity proof roots, a local cell stabilizer
`c(x) = -K_p(p-p_j)-K_v v`, and goal-independent viable successor options. The reference action
used to construct each frozen kappa return chain is not used as the normal Generator center. The
runtime searches diagonal Generator scales up to the actuator room and accepts only a candidate
that passes the unchanged complete `A_safe` and `A_cont` verifiers. Thus the center stabilizes a
local proof cell but does not encode a task or atlas traversal direction.

This does not prove task completion. `TASK_CONTROL_AUTHORITY_GATE` is an empirical software
diagnostic comparing center-only, random-in-Generator, and a goal-aware best-in-Generator oracle on
the same certified support. Its result is learnability evidence only; T_REC and T_AUTH continue to
depend on complete-set inclusion, not oracle performance.

### Physical density and temperature-target audit

The affine-tanh physical density remains
`log pi_a = log pi_u - log J_tanh - log|det G|`; no term is removed from the actor or Bellman
objective. Automatic-temperature design is a separate question because a fixed target entropy of
`-3` applied to `log pi_a` changes when certified physical support is rescaled even if the
normalized latent policy is unchanged.

Two candidate semantics remain for a controlled future comparison. **Option A** keeps a
physical-action entropy target but makes the target explicitly state/support dependent so its units
match `G(x)`. **Option B** keeps the complete physical density in actor and Bellman terms but adapts
temperature using normalized-support density `log pi_eta = log pi_u - log J_tanh`. Option B is
invariant to affine support scaling; it changes only temperature adaptation, not T12 density,
T_REC, T_AUTH, runtime authority, or executed-action critic semantics. No superiority claim is made
before a controlled experiment.

### Backup-event reward and normalized-temperature candidate

The persistent performance reward charges backup intervention once at the authority-transfer
event,
\[
r_{\mathrm{backup},t}=-\lambda_F\mathbf 1\{A_t\ne\mathrm{KAPPA\_BACKUP},\ A_{t+1}=\mathrm{KAPPA\_BACKUP}\}.
\]
Continuation under kappa does not repeat this event cost, but every recovery step still incurs the
ordinary elapsed-time and conservative realized-energy terms. Hence backup duration is not free,
and no separate backup-occupancy penalty is introduced.

For the controlled temperature candidate, define
\[
\log\pi_\eta=\log\pi_u-\log J_{\tanh},\qquad
\log\pi_a=\log\pi_\eta-\log|\det G|.
\]
T12, the actor objective, and the Generator Bellman branch continue to use the complete physical
density `log pi_a`. Only automatic temperature adaptation changes: the normalized candidate uses
`log pi_eta + H_target_eta` with `H_target_eta=-3`. Equivalently, it uses the state-dependent
physical target
\[
H_{\mathrm{target}}^a(s)=H_{\mathrm{target}}^\eta+\log|\det G(s)|,
\]
because `log pi_a + H_target_a = log pi_eta + H_target_eta`. This alpha residual is invariant to
uniform affine support scaling. It is an algorithm candidate, not a new performance theorem; all
safety, support, runtime-authority, and physical-density statements remain unchanged.

### Actor-gradient transmission audit boundary

A global critic ordering such as (Q(s,a_{oracle})>Q(s,a_{actor})) does not imply that the local
deterministic policy gradient is useful. The software audit therefore separates the chain
\[
\nabla_aQ
\;\longrightarrow\;
G^\top\nabla_aQ
\;\longrightarrow\;
\operatorname{diag}(1-\tanh^2u)G^\top\nabla_aQ
\;\longrightarrow\;
\nabla_\theta J_\pi.
\]
It compares the local autograd derivative with a centered finite difference toward the certified
best-in-Generator oracle, decomposes actor parameter gradients into exploitation and physical-density
entropy terms, and measures actor and critic Jacobians with respect to goal features. A frozen-critic
Q-only actor update is an offline diagnostic of gradient actionability; it never uses oracle labels,
never changes replay rewards, and is not a proposed control algorithm. These are optimization
diagnostics only and add no safety or convergence theorem.

### Counterfactual goal critic-preference boundary

For a fixed physical and certificate state (x), goal-dependent value is not the same claim as
goal-conditioned control preference:
\[
\frac{\partial Q(x,g,a)}{\partial g}\ne 0
\quad\not\Rightarrow\quad
\frac{\partial}{\partial g}\operatorname*{arg\,max}_{a\in C_{\rm run}(x)}Q(x,g,a)\ne 0.
\]
The counterfactual audit therefore holds (x), (C_{\rm run}(x)), (c), (G), recovery evidence,
and certificate identities fixed while replacing only the formally named goal-derived observation
fields. It compares value offsets, the action-gradient field, a searched critic-preferred certified
action, opposite-goal reversals, and alignment with the one-step environment oracle. The search is
diagnostic only: it does not supervise either network, enter replay, or modify rewards. These results
are optimization and representation evidence, not a new safety theorem or a global critic-optimum
claim.

### Bellman goal-action coupling and replay identifiability

The current empirical failure hypothesis is an almost additive critic,
\[
Q(x,g,a)\approx V(x,g)+A(x,a),
\]
rather than the task-control interaction required for goal-conditioned action preference,
`A=A(x,g,a)`. A goal-dependent value offset, or a nonzero local goal Jacobian, does not establish
that the preferred certified action changes with the goal. For the ReLU critic, a zero mixed second
derivative inside one activation region is likewise not primary evidence of goal insensitivity.

The Bellman coupling audit freezes physical/certificate state and complete Generator support,
propagates each certified candidate action through the same nominal plant convention, recomputes
the formal task reward for multiple counterfactual goals, and decomposes
\[
Y_{ij}=R_{ij}+\gamma Q_{\rm target}(s'_{ij},a'_{ij})
-\gamma\alpha\log\pi_a(a'_{ij}\mid s'_{ij}).
\]
It compares the centered goal-action interaction in `R`, `Y`, and learned `Q`, plus the finite-set
preferred actions. The target actor uses common reparameterization noise across goals for each
fixed physical successor so goal comparisons are not contaminated by sampling noise. Completion
transitions are separated because the continuing MDP assigns a fresh next goal.

Replay identifiability is audited independently in local physical neighborhoods using recovery
cell, position, velocity, energy, and mode. Goal angular coverage, normalized Generator-action
coverage, and the rank/conditioning of `g tensor-product eta` are reported. Counterfactual goal
augmentation is diagnostic only: it never enters replay or updates a network. These diagnostics
add no safety theorem and do not implement n-step returns, relabeling, or a new critic.

### Crossed horizon, goal coverage, and soft-entropy audit

The crossed audit separates immediate task-control reward, return horizon, replay goal coverage,
physical soft entropy, and critic representation capacity without updating the production algorithm.
For a valid fixed-goal segment it evaluates
\[
Y_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k}
+\gamma^n\left[Q_{\rm target}(s_{t+n},a'_{t+n})
-\alpha\log\pi_a(a'_{t+n}\mid s_{t+n})\right].
\]
The physical-density contribution is separated exactly as
\[
-\alpha\log\pi_a=-\alpha\log\pi_\eta+\alpha\log|\det G|.
\]
The no-entropy and normalized-entropy targets localize preference sources only; neither is a training
proposal and the physical SAC density theorem is unchanged.

Large target contrast is not useful goal-conditioned control evidence when goal-generic bootstrap
value or state-dependent entropy/support volume dominates it. Likewise, cumulative reward growth is
not horizon restoration unless preferred certified actions gain opposite-goal reversal, oracle
alignment, and nonadditive interaction. A temporary same-architecture supervised critic may fit fixed
audit targets after per-state-goal action-advantage standardization. It is isolated from production
networks and tests representation capacity only.

### Goal-radius completion boundary and exposure prerequisite

For normal task authority, completion uses the pre-transition pending goal and the closed set
\[
\|p_{t+1}-g_t\|\le r_g.
\]
The software comparison includes a fixed numerical tolerance solely to represent this closed-set
boundary consistently. Entering the radius during backup or charging does not complete a task.
This is a task-semantic contract, not a new safety theorem.

Persistent evaluation assigns a new goal only after completion. Consequently, a run with no
completion exposes its network to only one task goal. Such a run cannot establish general
goal-conditioned learning failure. The controlled training collector may therefore terminate a
rollout after a configured exposure interval and resample `(x,g)` without resetting the policy,
critics, target critics, temperature, optimizers, gradient counter, or replay. The final physical
transition receives a distinct collector-boundary no-bootstrap mask, so the one-step Bellman target
never uses the unrelated reset state. This changes data collection only; persistent evaluation and
all task-independent safety definitions remain unchanged.
