# Single-UAV Certification Experiment Environment

## Status and Claim Boundary

This package is a deterministic synthetic experiment environment. It exercises proof-object generation, complete-set propagation, watchdog fallback, affine-tanh action execution, replay semantics, and legacy compatibility. Its calibration records have `evidence_kind="synthetic-simulator"`; they are not real sensor, dynamics, energy, terminal, WCET, atomic-I/O, HIL, or flight evidence. The strongest mathematical conclusion remains corridor-conditional and conditional on independently valid physical contracts.

## Legacy Isolation

The original implementation remains at `envs/UAVEnergyDelivery.py`, so existing imports and training scripts retain their behavior. `envs/legacy/multi_uav_delivery_env.py` is an explicit alias namespace for that unchanged empirical multi-UAV environment. Its order assignment, communication, collision correction, recharge/respawn, and multi-station semantics are not copied into the certified environment.

## Package Structure

```text
envs/certified_uav/
  config.py          synthetic configuration and braking-range gate
  state.py           physical state and shape validation
  scenario.py        deterministic JSON scenario loading and static checks
  obstacles.py       ground-truth static world and swept collision
  dynamics.py        sole double-integrator implementation
  actuator.py        actuator box, bounded tracking, and ActionTrace
  lidar.py           32-ray LidarPacket and ground-truth measurement model
  energy.py          realized simulation energy model
  terminal.py        hover-evidenced terminal predicate
  plant_env.py       physical Gymnasium plant accepting only a_exec
  task_wrapper.py    task observation and reward only
  runtime_wrapper.py certificate state, closure, generator, watchdog, replay
  telemetry.py       step telemetry and synthetic calibration records
  adapters.py        state/LiDAR adapter interfaces
  scenarios/         open, narrow, invalid, and insufficient-energy fixtures
```

## Layering

`CertifiedSingleUAVPlantEnv` receives only a final three-dimensional acceleration in the actuator box. It applies bounded tracking, calls `integrate_double_integrator`, performs swept capsule collision, subtracts realized energy, measures LiDAR, evaluates the terminal predicate, and terminates on collision, energy depletion, velocity-limit violation, or terminal success. It does not construct κ, c, G, certificates, actors, task goals, or fallback decisions.

`CertifiedTaskWrapper` owns the mission phase, goal, and task reward. Its fixed 121-dimensional observation layout is:

| Segment | Shape | Slice |
|---|---:|---:|
| normalized position | 3 | 0:3 |
| normalized velocity | 3 | 3:6 |
| normalized energy | 1 | 6:7 |
| normalized goal delta | 3 | 7:10 |
| normalized station delta | 3 | 10:13 |
| normalized LiDAR distance | 32 | 13:45 |
| LiDAR validity | 32 | 45:77 |
| task-only local-map encoding | 16 | 77:93 |
| task-only corridor encoding | 24 | 93:117 |
| mission-phase one-hot | 4 | 117:121 |

These task features are never certificate evidence. `MissionPhase` is explicit task state with
`OUTBOUND`, `RETURN`, `SUCCESS`, and `FAILURE`; it is recorded for learning and metrics but does
not replace any geometric, recovery, or energy proof object.

`CertifiedRuntimeWrapper` consumes only physical state and `LidarPacket` on its certificate path. A synthetic reset may replay scenario-provided LiDAR packets to represent historical rolling-map evidence; the runtime never reads the plant obstacle list. Each execution cycle stages the previously authorized κ or emergency brake before certificate work, updates the proof-carrying ternary grid, runs `SingleCorridorClosurePipeline`, checks recovery and E3 certificates, constructs the complete diagonal zonotope, calls the actor only after set acceptance, and uses one-shot watchdog publication. The plant receives only the resulting `a_exec`.

The Python cycle profiler is not hard WCET evidence. The wrapper reports `wcet_status="blocked-by-deployment-evidence"`; therefore the synthetic fixture permits profiled certificate construction while the independent actor/candidate watchdog still enforces its configured deadline. A deployment-qualified WCET contract is required before interpreting the complete cycle as real-time certified.

## Authoritative Dynamics and Actuation

All plant and calibration-residual predictions call:

```python
p_next = position + dt * velocity + 0.5 * dt * dt * measured_acceleration
v_next = velocity + dt * measured_acceleration
```

The declared actuator set is the coordinate box `[-a_max, a_max]^3`. Box corners are accepted without Euclidean normalization. Out-of-box publication is an interface error; the plant does not clip it. `ActionTrace` separately records actor latent/nominal output, mapped candidate, fallback, published action, measured action, acceptance, fallback reason, and certificate epoch.

## Collision, Energy, LiDAR, and Terminal Semantics

- Collision uses the full line segment from the previous to next position against body-inflated world boundaries, AABBs, and cylinders. A collision terminates; no safe-position correction or velocity reset occurs.
- Simulation energy is `dt*(c0 + c_v|v| + c_a|a_measured|^2 + c_compute + c_comm) + epsilon_sim`. Depletion terminates without respawn or recharge. The simulator cost and certificate `cost_upper` remain distinct.
- `LidarPacket` contains 32 distances, validity flags, hit flags, angles, timestamp, measured pose, heading, and sensor version. Valid no-hit, valid hit, and invalid are distinct. Existing strict grid logic does not promote invalid or maximum-range no-hit rays to FREE.
- `TerminalSpec` requires position, velocity, minimum energy, and evidenced hover continuation. Docking and charging handoff are not claimed.

## Fixed Scenarios

- `open_corridor.json`: four overlapping AABB levels, one-step vertical braking descent, complete synthetic manifest, and positive-volume generator.
- `narrow_corridor.json`: inherits the open fixture but requires `sigma_G=0.06`, exceeding one actuator half-width; recovery remains available while generator construction returns `NO_GENERATOR_SET`.
- `invalid_corridor.json`: breaks the suffix overlap/initial containment and is rejected before task execution.
- `insufficient_energy.json`: starts below terminal-plus-reserve energy and disables task mode.

These parameters are deterministic software fixtures, not calibrated UAV values. `RandomTrainingScenario` intentionally remains unimplemented.

## Generator-SAC Integration

The runtime replay record stores the certificate snapshot, task observation, u, tanh(u), c, G, candidate, κ, executed action, acceptance, fallback reason, hashes, versions, timestamp, and measured tracking action. `CertificateEpoch` is emitted from the immutable snapshot. `GeneratorSACTrainer` rejects cross-epoch batches, detaches c and G, feeds `a_exec` to the critic, applies the stable tanh Jacobian and `-log|det G|` only on accepted generator transitions, and excludes fallback atoms from the generator entropy density.

The formal multi-step implementation is `cert_runtime/generator_sac.py`. It adds twin online and
target critics, discounting, terminal masking, branch-aware next actions, automatic temperature,
and Polyak updates. For a Generator-valid next state it uses

```text
y = r + gamma (1-d) [min(Q1_target,Q2_target)(o_next,a_next)
                      - alpha log q_G(a_next|z_next)].
```

For a fallback-only next state it instead evaluates `kappa(next_state)` and adds no Generator
entropy. Current critics always receive replayed `a_exec`. Actor updates use only accepted
Generator rows; fallback-only batches explicitly skip the actor objective. The first training
profile groups replay by certificate epoch and makes no monotone-improvement claim across epoch
changes.

## Multi-Step Mission Fixtures

### Baseline audit before this phase

| Limitation | Actual pre-phase code | Audit result |
|---|---|---|
| reset after every training step | `scripts/train_generator_sac_smoke.py` unconditional `runtime.reset` | confirmed |
| independent one-step replay | smoke transition followed immediately by reset | confirmed |
| episode length/return hard-coded | `episode_length=1`, `episode_return=reward` | confirmed |
| near-terminal training fixture | `open_corridor` starts adjacent to the terminal | confirmed; retained only for acceptance |
| immediate-reward critic | `MinimalGeneratorSAC` regressed Q directly to reward | confirmed; smoke control retained |
| missing SAC bootstrap/targets | no gamma, target critics, masks, or Polyak update | confirmed; formal trainer added separately |
| reset-based stress | stress loop reset each cycle | confirmed; remains software stress only |
| no baseline runner | no common agent/experiment registry | confirmed; `experiments/` added |

The acceptance fixtures remain unchanged. Training uses four additional deterministic scenarios:

- `mission_open`: short outbound inspection followed by a long return to the terminal;
- `mission_obstacle`: static blocked direct route with an explicit synthetic detour region;
- `mission_narrow`: a narrow certified region where positive-volume Generator sets may disappear;
- `mission_energy_tight`: an early energy-triggered return regime.

The training certificate profile consists of explicitly supplied synthetic free/occupied boxes and
return waypoints. `SyntheticMissionCertificateProvider` propagates the complete diagonal action
interval for one step, checks actuator/velocity/free-region/occupied-region containment, and
constructs a versioned Generator bundle. This is a fast synthetic training fixture, not a substitute
for the proof-carrying online LiDAR/corridor closure used by the acceptance scenarios and not real
calibration evidence.

The deterministic `mission_open` controller regression completes outbound and terminal return in
more than 20 consecutive plant steps without reset. Training loops reset only on `terminated` or
`truncated`; episode return, length, task completion, return trigger, collision, depletion, and
timeout are accumulated per episode.

## Unified Baselines

`experiments/` provides four execution modes over the same plant, task reward, scenario seed,
network width, replay budget, and optimizer family:

1. `sac`: direct coordinate-box SAC without certificate execution;
2. `penalty_sac`: direct SAC with additional sampled failure penalties;
3. `shield_sac`: nominal SAC action executed only when it belongs to the verified runtime set,
   otherwise κ; its critic receives the post-shield action;
4. `generator_sac`: affine-tanh Generator branch with κ fallback and branch-aware entropy.

The baselines compare empirical execution behavior. Penalty success is not a certificate, and a
shielded zero-collision sample does not establish a continuous-domain theorem.

## Synthetic Calibration Log

`export_calibration_record()` returns structured records containing timestamps, states, commanded/candidate action, published action, measured action, next state, LiDAR arrays, energy before/after, and sensor/dynamics/tracking/energy/terminal versions. Voltage, current, and power remain optional because this environment does not synthesize an electrical measurement chain. These logs can test calibration software but cannot become deterministic engineering bounds merely because simulated residuals are covered.

## Verification Commands

```bash
.venv/bin/python -m unittest discover -v -s tests -p 'test_*.py'
```

The acceptance-round unittest result is 100 passed, 0 skipped, and 0 failed in 131.893 seconds (128.74 seconds wall time) under Python 3.12.3, NumPy 1.26.4, and Torch 2.7.1+cu128. The tests validate software semantics and deterministic fixtures; they do not validate real flight safety. `unittest` remains authoritative and pytest was not introduced.

## Remaining Blockers

- Real sensor, localization, timing, dynamics, tracking, wind, energy, and terminal calibration remains blocked-by-calibration.
- The synthetic LiDAR bootstrap is a deterministic historical-packet fixture, not SLAM or a deployment map-maintenance proof.
- Desktop Python timing is profiling only; independent RTOS scheduling, WCET, and atomic actuator-bus evidence remain blocked-by-deployment-evidence.
- The fixed corridor uses a deliberately simple one-step vertical braking hierarchy to close software proof dependencies. A physically meaningful return flight corridor still requires calibrated cell construction and recovery verification.
- HIL, command readback, power instrumentation, and real actuator tracking are unresolved.
- No global unknown-environment safety, automatic docking, charging handoff, dynamic obstacle, multi-UAV, or large-scale training claim is made.

## Mission Certificate Gate and Comparative Validation

The one-step mission fixture was replaced by a corridor-wide synthetic recovery manifest. For every
task-tube root it contains a finite strict-descent chain to the terminal set, complete correlated
state-tube propagation, actuator/velocity bounds, swept FREE-union geometry, robust energy recursion,
E3 residuals, and hash-linked dependencies. The mandatory gate passed for all four mission scenarios.
The strengthened verifier checked 86,684 recovery cells in total; 48 sampled chain rollouts were used
only as debugging evidence and recorded zero sampled collision, level, energy, or terminal-arrival
failures. These bounds remain synthetic and are not physical calibration evidence.

The repaired 2,000-step matrix contains 48 runs. Both certified methods recorded zero sampled
collisions, zero uncertified task publications, and zero fallbacks with invalid κ. In
`mission_narrow`, κ remains certified while the state-uncertainty box intersects a conservative
Generator-exclusion region; the observed `NO_GENERATOR_SET` rate is 0.005 and recovery still reaches
the terminal. This status means that this constructor produced no verified full-rank set, not that
no mathematical feasible set can exist.

The formal validation contains four methods, four scenarios, five seeds, 10,000 environment steps,
1,000 warmup steps, batch size 128, and 20 deterministic evaluation episodes per seed. The complete
mean/std table is `artifacts/comparison/aggregate/summary.csv`; raw per-episode rows, evaluation rows,
safety events, compact learning curves, runtime data, and safety-performance data are stored beside
it. Selected training means are:

| Method | Scenario | Task success | Return success | Collision rate | Fallback rate |
|---|---|---:|---:|---:|---:|
| Generator-SAC | open | 1.000 | 1.000 | 0.000 | 0.9055 |
| Generator-SAC | obstacle | 1.000 | 1.000 | 0.000 | 0.6710 |
| Generator-SAC | narrow | 0.000 | 1.000 | 0.000 | 0.7600 |
| Generator-SAC | energy-tight | 0.000 | 1.000 | 0.000 | 0.9342 |
| Shield-SAC | open | 0.000 | 1.000 | 0.000 | 0.9900 ± 0.0014 |
| Shield-SAC | obstacle | 0.000 | 1.000 | 0.000 | 0.9825 ± 0.0007 |
| SAC | open | 0.0278 ± 0.0160 | 0.000 | 0.2424 ± 0.2665 | N/A |
| Penalty-SAC | obstacle | 0.000 | 0.000 | 0.2349 ± 0.0595 | N/A |

Generator-SAC lowers intervention relative to Shield-SAC in every fixture, by about 8.45 percentage
points in open, 31.15 in obstacle, 22.00 in narrow, and 5.67 in energy-tight. The task-oriented
Generator completes task and return in open/obstacle, but the narrow exclusion and energy trigger
intentionally cause early certified return before task completion. All 20 Generator runs performed
8,998--9,000 finite actor updates and 9,000 critic updates. Identical task/return rates across seeds
show that the synthetic certificate geometry and deterministic evaluation dominate this 10k study;
the result validates mechanism semantics rather than establishing learning convergence.

Runtime values are desktop profiling. Direct methods report certificate timing as N/A, never zero.
Generator total p99 is approximately 15.9 ms (open), 23.2 ms (obstacle), 23.1 ms (narrow), and
14.8 ms (energy-tight); these measurements are not hard WCET or RTOS evidence.

Reproduce the checked-in reduced matrix with:

```bash
.venv/bin/python scripts/run_comparison.py \
  --methods sac penalty_sac shield_sac generator_sac \
  --scenarios mission_open mission_obstacle mission_narrow mission_energy_tight \
  --seeds 0 1 2 --steps 2000 --warmup-steps 200 --batch-size 64 \
  --evaluation-episodes 20 --output-dir artifacts/comparison_2k_closed
```

The formal command is the same with seeds `0 1 2 3 4`, 10,000 steps, 1,000 warmup steps, batch size
128, and output directory `artifacts/comparison`. Parallel scenario workers may use
`--skip-aggregate`; `experiments.aggregate.aggregate_results` is run once after all workers finish.
The final post-closure authoritative suite ran 119 tests in 160.314 seconds: 119 passed, zero
failed, and zero skipped under the UV-managed environment.

## Mission certificate closure and diagnosis

The former `SyntheticMissionCertificateProvider` checked only a fast one-step task successor and
could not support recursive feasibility.  `MultiStepSyntheticMissionCertificateProvider` now
builds a levelled, hash-linked recovery chain for every synthetic task-tube root.  Every complete
cell verifies the full correlated state tube, tracking/timing/model disturbance, actuator and
velocity bounds, swept FREE-union geometry, strict lower-level descent, robust energy recursion,
and terminal linkage.  `scripts/validate_mission_recovery_certificate.py` is the mandatory gate;
training aborts with `blocked-by-mission-certificate` when any declared cell fails.

Generator and Shield SAC use the same frozen certified κ.  They differ only in task-action
parameterization: Shield verifies a nominal point action against the task successor predicate,
whereas Generator verifies the complete full-rank affine set before sampling.  A missing Generator
set does not invalidate κ.  Conversely, a valid Generator set is not evidence that κ is certified.

The earlier zero task-success result had two implementation causes.  First, the purported
task-oriented center used the recovery-chain root action, which points toward the station.  It now
uses the explicit outbound task reference and is independently re-verified as a complete zonotope.
Second, an 80 ms desktop watchdog window caused irreversible recovery under Python scheduling; the
synthetic mission fixture now uses 150 ms, below its 200 ms control period.  This remains desktop
profiling, not hard-WCET evidence.

The deterministic center ablation in `artifacts/generator_center_ablation/results.csv` shows that
the verified braking center stays 0.8 m from the task for 400 steps, while the verified
task-oriented center completes the task and certified return in 226 steps for all three fixture
seeds.  This supports a performance-design choice only: the complete set verifier, not the task
proposal, remains the certificate source.
## Acceptance and Generator-SAC smoke protocol (2026-08-07)

This environment is a synthetic software fixture. Its strongest supported statement is a
**corridor-conditional software invariant under synthetic bounds**; it is not real-flight,
calibration, HIL, atomic-I/O, or hard-WCET evidence.

## Code-structure acceptance audit

| Requirement | Actual file/function | Verified call path | Evidence | Result |
|---|---|---|---|---|
| Legacy alias remains usable | `envs/legacy/multi_uav_delivery_env.py` | alias to `envs.UAVEnergyDelivery` | legacy reset/step regression | software-verified |
| Plant receives only `a_exec` | `plant_env.py:CertifiedSingleUAVPlantEnv.step` | runtime publisher -> task wrapper -> plant | action-trace/replay equality tests | software-verified |
| Plant has no actor/certificate access | `plant_env.py` | no runtime imports or calls | source/call audit | software-verified |
| Certificate map has no obstacle-list access | `runtime_wrapper.py` | `LidarPacket.to_certificate_rays` -> ternary grid | leakage regression | software-verified; bootstrap packets are synthetic fixtures |
| One dynamics implementation | `dynamics.py:integrate_double_integrator` | plant, runtime predictor fixture, calibration residual | unified-dynamics tests | software-verified |
| No action-norm rescaling or hidden action clipping | `actuator.py:validate_action_box` | reject outside coordinate box | box-corner tests | software-verified |
| No collision repair or energy respawn | `plant_env.py:step` | terminal transition retains physical result | swept-collision/depletion tests | software-verified |
| Versioned 32-ray packet | `lidar.py:LidarPacket` | simulator -> task packet/certificate rays | packet tests | synthetic-validated |
| Task/certificate state separation | `task_wrapper.py`, `runtime_wrapper.py` | task features never enter proof objects | reward-isolation test | software-verified |
| Kappa staged before actor | `runtime_wrapper.py:step` | `stage_default` before `prepare_certificate_cycle` and worker | watchdog trace | software-verified |
| Invalid certificate bypasses actor | `runtime_wrapper.py:step` | fail branch publishes staged kappa | scenario matrix | software-verified |
| Immutable/versioned bundle | `watchdog.py:CandidateBundle` | complete bundle -> watchdog recheck | stale/tamper tests | software-verified |
| One-shot publication | `watchdog.py:AtomicCommandPublisher` | exactly one `publish_once` succeeds | watchdog/stress tests | software-verified |
| Replay separates actions | `runtime.py:ReplayRecord`, `actuator.py:ActionTrace` | candidate/fallback/executed/measured fields | replay tests | software-verified |
| Critic consumes `a_exec` | `trainer.py`, `smoke_training.py` | replay `executed_action` -> critic | Torch tests | software-verified |
| Fallback excludes Generator entropy | `smoke_training.py:MinimalGeneratorSAC.update` | accepted-index branch only | fallback-only test | software-verified |
| Frozen optimization epoch | `trainer.py:CertificateEpoch` | batch validation before update | mixed-epoch rejection | software-verified |

The original 50 ms synthetic watchdog setting was scheduler-flaky on desktop Python. The
synthetic fixture now uses 100 ms (still below the 200 ms control period). This is a test
stability setting, not a deployed WCET claim; explicit zero-deadline injection still verifies
fail-closed behavior.

## Reproducible acceptance commands

```bash
.venv/bin/python -m unittest discover -v -s tests -p 'test_*.py'
.venv/bin/python scripts/run_certified_env_acceptance.py --scenario open_corridor --seed 0 --matrix
.venv/bin/python scripts/run_certified_env_stress.py --seeds 0 1 2 3 4 5 6 7 8 9
.venv/bin/python scripts/train_generator_sac_smoke.py --scenario open_corridor --seeds 0 1 2
```

Artifacts are written under `artifacts/environment_acceptance/` and
`artifacts/smoke_training/`. The stress run is software-error discovery only. The training run
freezes the certificate epoch, rebuilds no corridor online, trains critics on `a_exec`, applies
the affine-tanh density only to accepted transitions, and separately records fallback atoms.

The authoritative smoke artifact contains three seeds, 1,000 environment steps and 97 gradient
steps per seed. Every seed recorded 1,000 accepted transitions, zero fallback transitions, zero
nonfinite values, and nonzero actor parameter change. The 500-cycle stress artifact recorded zero
candidate-membership, executed-action, one-shot publication, synthetic energy-bound, collision,
or depletion violations. These are synthetic software observations, not physical guarantees.

The accepted open-corridor cycle uses the already closed, frozen certificate epoch and completed
in 71.54 ms on the recorded desktop run; its task-worker watchdog interval remained below the
100 ms synthetic deadline and publication occurred once. The initial online synthetic closure is
performed before that cycle. A separate online rebuild profile took approximately 0.87 s, dominated
by the Python ternary-grid update, so online rebuild is not deployment-qualified. In the frozen
500-cycle profile, total-cycle median/p95/p99/max were approximately 69.8/76.7/84.2/93.7 ms and
one-shot publication itself was measured in microseconds. These values are profiling, not WCET.

The pre-modification baseline audit did not reproduce the claimed 92/92 result: it recorded 91
passes and one scheduler-sensitive open-corridor failure under the former 50 ms desktop watchdog
window. Raising the synthetic fixture window to 100 ms removed that test nondeterminism while the
explicit zero-deadline test continued to fail closed. The final authoritative suite records 100/100.

## Status boundaries

- **Software-verified:** action plumbing, bundle immutability, one-shot publication, replay
  semantics, epoch rejection, affine-tanh formula, and fail-closed software injections.
- **Synthetic-validated:** scenario closure, collision/energy fixtures, and stress statistics.
- **Empirical-training-only:** finite losses and parameter updates in the smoke run.
- **Blocked-by-calibration:** physical sensor, dynamics, tracking, energy, and terminal bounds.
- **Blocked-by-deployment-evidence:** RTOS scheduling, hard WCET, atomic command bus, and hardware
  action readback.
- **Unresolved:** transfer from synthetic corridor closure to real rolling geometry and HIL.
- **Theory-only:** global convergence or monotonic improvement across changing certificate epochs.

## RL Contribution and Scenario Families

`center_only`, `random_generator`, and `generator_sac` execute through the same certified runtime.
Center-Only publishes the verified center when a Generator exists and κ otherwise.
Random-in-Generator samples `u ~ N(0,I)` without training and publishes `c + G tanh(u)`; membership
comes from complete-zonotope verification, not random testing.

Metrics separate outbound intervention from planned RETURN handoff, `NO_GENERATOR_SET`, certificate
failure, and deadline failure. The previous overall fallback rate was dominated by planned recovery
and must not be interpreted as an RL failure rate.

Deterministic scenario families are materialized under `artifacts/scenario_families/` with disjoint
training, validation, and held-out IDs. Each file carries scenario and geometry hashes; every usable
file must independently produce a PASS manifest whose hash is stored in the index. Start, task,
energy, bounded tracking disturbance, and conservative obstacle realization can vary only inside
declared synthetic domains. A 110% disturbance case invalidates the certificate and is reported as
an out-of-contract diagnostic. None of these tests constitutes physical calibration evidence.

### Current ablation result

The five-seed task-oriented-center ablation records success/return 1.0 and sampled collision 0 for
Center-Only, Random-in-Generator, and Generator-SAC on both open and obstacle fixtures. Center-Only
requires 226 steps in open and 451 in obstacle. Generator-SAC differs by 0 and -0.15 mean steps,
respectively; path and energy differences are below 0.002 synthetic units. Hence current task
success is predominantly produced by the verified center. Zero and braking centers do not approach
the task in either fixture, despite passing the same complete-set verifier; center construction is
a performance mechanism, while verification remains the certificate source.

### Held-out scenario-family result

The deterministic family generator produced 20 training, 10 validation, and 20 held-out synthetic
scenarios. All 50 independently rebuilt manifests passed their synthetic mission-certificate gate,
and all scenario IDs, scenario hashes, and manifest hashes are distinct and split-disjoint. The
frozen-policy evaluation uses one available 10k checkpoint seed and 20 episodes for each held-out
scenario; it is therefore a single-checkpoint-seed pilot rather than a multi-seed learning claim.

On the five held-out open and five held-out obstacle scenarios, Generator-SAC and Center-Only both
obtain task/return success 1.0, zero sampled collision, and zero OUTBOUND intervention. Shield-SAC
returns successfully with zero sampled collision but task success 0, with mean OUTBOUND
intervention 0.40 in open and 0.133 in obstacle. In narrow scenarios, Generator-SAC and Center-Only
retain return success 1.0 with task success 0 and about 1.97% OUTBOUND intervention; this is the
intended `kappa valid / no positive-volume Generator` behavior. In energy-tight held-out scenarios,
Generator-SAC and Center-Only achieve task success 0.20 and return success 1.0. The identical
center/actor result shows that this loss is driven by task/energy-trigger semantics rather than
actor memorization. Across all 60 method-scenario rows there are zero sampled certified-method
collisions, zero uncertified task publications, and zero invalid-kappa fallbacks.

`RL_CONTRIBUTION_GATE` and `GENERALIZATION_GATE` both pass their software-evidence criteria. They do
not imply physical calibration, hard WCET, or global safety. A 50k multi-scenario trainer is
implemented but was not run in this round: the RL ablation shows negligible residual benefit, while
the held-out failures are shared by Center-Only and Generator-SAC. Increasing actor training before
changing the center/task-trigger challenge would not isolate an RL contribution.

### Sensitivity and certificate scale

The in-contract disturbance sweep from 0% through 100% of the declared synthetic bound preserves
the open-fixture result; 110% is rejected as out of contract. Bound-scale factors 0.5, 0.75, and 1.0
produce the same Generator volume in this fixture, showing that another configured limit is active;
this sweep does not yet demonstrate a robustness-performance Pareto frontier.

The four manifests contain 86,684 recovery cells. Desktop construction takes approximately
5.3--5.5 s for each 8,221-cell manifest, 25.8 s for the 37,802-cell obstacle manifest, and 22.5 s
for the 32,440-cell narrow manifest. Serialized sizes are 7.9--36.6 MiB. Current root lookup is a
linear scan followed by exact containment and averages 1.7--3.5 ms in the recorded desktop profile.
These are offline/profile measurements, not online WCET evidence; a spatial index may accelerate
lookup but cannot replace final cell-containment verification.

The authoritative post-change suite is `128 tests in 176.973 s`, with 128 passes, zero failures,
and zero skips. The four mission certificate gates also pass after regeneration. Exact raw outputs
are stored in `artifacts/paper/`.

## Persistent goal stream and charging extension

The persistent extension is isolated from all single-mission regression fixtures. It adds a finite
certified goal network, pending-goal preservation across charging, event-level energy management,
hard forced-return and departure-energy gates, and synthetic net charging. The environment assigns
the goals; the policy does not schedule or select them. Persistent configs set
`terminate_on_terminal=false`; single-mission configs retain the default `true`.

The low-level certificate chain is unchanged. The energy-management policy requests service versus
charging, the certificate runtime may override it, and kappa remains independently certified. Its
categorical entropy is not Generator affine-tanh entropy. See
`docs/PERSISTENT_TASK_CHARGING.md`. This round added code and deterministic unit tests only; no
persistent gate, acceptance rollout, baseline comparison, or energy-management training result is claimed.
