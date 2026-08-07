# Physical Calibration Replacement Plan

This plan maps each synthetic certificate parameter to future physical evidence. It does not
provide that evidence and does not upgrade any current result to real-flight safety.

| Physical quantity | Current synthetic contract parameter | Required acquisition | Bound semantics required for strict use | Invalidation trigger |
|---|---|---|---|---|
| Localization | `SensorCalibrationContract.position_error` | Ground-truth-referenced position logs over the certified domain | deterministic engineering envelope or simultaneous confidence set with stated delta | estimator, firmware, payload, or domain change |
| Scan attitude/direction | `attitude_error`, `beam_half_width` | calibrated target scans over heading and range | upper angular tube including synchronization | LiDAR mount or time-source change |
| LiDAR range | `range_error`, `maximum_reliable_range` | independent calibration/validation targets | no empirical quantile relabelled deterministic | optics, firmware, environment domain, expiry |
| Action tracking | `TrackingCalibrationContract.acceleration_error` | commanded, published, measured actuator-aligned logs | componentwise simultaneous outer bound over velocity/payload modes | controller or actuator change |
| Dynamics residual | `DynamicsCalibrationContract.position_residual`, `velocity_residual` | synchronized state transitions using the authoritative double-integrator residual | one-step outer envelope including latency and floating-point budget | dynamics software, payload, control period change |
| Wind/disturbance | `wind_acceleration_radius` | controlled fan/tunnel and field residual data | bounded disturbance set or explicitly probabilistic theorem | weather domain exceeded |
| Energy cost | `EnergyCalibrationContract` power coefficients and underestimation margin | aligned voltage/current/power, state, and executed-action logs | full-cell/full-action one-step upper cost, not expected cost | battery, temperature, payload, compute stack change |
| Terminal admissibility | `TerminalCondition` position, velocity, energy, continuation modes | hover/descent/docking continuation trials and independent validation | set containment plus minimum reserve | terminal geometry or continuation controller change |
| Runtime timing | `WCETContract` stage budgets | target CPU/RTOS, affinity, command-bus and worst-input measurements | deployment-qualified WCET evidence | platform, build, thread policy, workload change |
| Command readback | `PublishedCommand` and tracking log | atomic command bus timestamps and actuator readback | one-shot ordering and stale-command rejection evidence | bus/firmware change |

## Acquisition sequence

1. Freeze device, firmware, controller, payload, battery, and operating-domain identifiers.
2. Collect training/calibration/validation partitions with immutable evidence IDs and timestamps.
3. Estimate candidate envelopes from calibration data only.
4. Validate simultaneous coverage on the untouched validation partition and record exceedances.
5. Populate versioned contracts without unsourced default flight values.
6. Rebuild the complete corridor, recovery, energy, and Generator manifests.
7. Qualify timing and atomic publication on the deployment platform.
8. Run HIL fault injections; treat them as empirical evidence, not mathematical proof.

Until these steps are complete, sensing, dynamics, tracking, energy, and terminal premises remain
`blocked-by-calibration`, while timing and atomic I/O remain `blocked-by-deployment-evidence`.
