# Frozen SAC baseline and next experiment tracks

## Frozen reference

`artifacts/phase1_sb3_sac_1m_gpu/` is the immutable reference artifact for
`DIRECT_SAC_BASELINE_SOLVED`. Its final held-out rate is approximately 26--27
tasks per 1,000 steps with a median of approximately 37 steps per completed
goal. New launchers never target this directory.

## Track A: energy and charging exploration

Track A asks whether one continuous Standard SB3 SAC policy can execute random
goals, voluntarily visit the fixed station at low energy, dwell to charge,
depart, and resume the same pending goal. It uses
`PersistentEnergyNavigationEnv`, no discrete charge/return action, no oracle
energy margin, no Generator, no kappa, and no learned safety field.

## Track B: untuned algorithm baselines

Track B runs Standard SB3 PPO and DDPG in the exact solved open navigation
environment. These runs are named `UNTUNED_STANDARD_SB3_BASELINES`. They are
algorithm references, not evidence for charging learnability and not a reason
to reopen the solved SAC conclusion.
