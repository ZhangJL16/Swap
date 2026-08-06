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

`CertifiedTaskWrapper` owns the inspection goal and task reward. Its fixed 117-dimensional observation layout is:

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

These task features are never certificate evidence.

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
