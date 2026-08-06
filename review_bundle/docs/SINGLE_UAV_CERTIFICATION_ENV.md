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

## First Multi-Step Comparison (Reduced Validation Budget)

The checked-in matrix runs four methods, four mission scenarios, three seeds, and 2,000 environment
steps per run (96,000 total transitions). This is deliberately below the planned 10k--20k first
paper-training budget and is therefore a pipeline/semantic validation, not a converged comparison.
All runs use 200 warmup steps, batch size 64, the same plant/task/scenario seeds, and the actual
executed action in each critic replay row.

| Method | Scenario | Task success | Return success | Collision episode rate | Fallback rate |
|---|---|---:|---:|---:|---:|
| Generator-SAC | energy-tight | 0.000 | 0.930 | 0.000 | 0.160 |
| Generator-SAC | open | 0.000 | 0.000 | 0.000 | 0.111 |
| Generator-SAC | obstacle | 0.000 | 0.000 | 0.000 | 0.094 |
| Generator-SAC | narrow | 0.000 | 0.000 | 0.000 | 0.087 |
| SAC | open | 0.024 | 0.000 | 0.607 | 0.000 |
| Penalty SAC | energy-tight | 0.049 | 0.000 | 0.613 | 0.000 |
| Shield SAC | open | 0.000 | 0.000 | 0.000 | 0.780 |
| Shield SAC | obstacle | 0.000 | 0.000 | 0.022 | 0.817 |

Values are means over three seeds. The full 16-row mean/std table is
`artifacts/comparison/aggregate/summary.csv`; learning-curve bins, safety-performance data, and
runtime data are adjacent CSV files. The energy-tight Generator result represents mostly
energy-triggered return, not outbound task completion. The zero task-success values for the main
method show that 2,000 steps are insufficient for a learning claim. The sampled shield-obstacle
collision demonstrates that the fast synthetic mission κ/waypoint profile is not yet a closed
corridor-wide recovery certificate. It must be repaired or replaced by the full closure pipeline
before using that scenario for theorem-facing safety evidence.

No sampled Generator-SAC collision occurred in these 48 reduced runs, but that observation is only
synthetic empirical evidence. It does not establish calibrated physical safety or continuous-domain
coverage.

Reproduce the checked-in reduced matrix with:

```bash
.venv/bin/python scripts/run_comparison.py \
  --methods sac penalty_sac shield_sac generator_sac \
  --scenarios mission_open mission_obstacle mission_narrow mission_energy_tight \
  --seeds 0 1 2 --steps 2000 --warmup-steps 200 --batch-size 64
```

The authoritative post-change suite contains 113 tests. It includes the original acceptance,
calibration, interval, watchdog, environment, and Torch tests plus 13 multi-step/SAC/fairness tests.
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
