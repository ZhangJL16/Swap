# Experiment Integrity Audit

**Date:** 2026-08-07  
**Evaluation type:** simulation_only  
**Verdict:** WARN (mechanism validation is supported; convergence and physical-safety claims are not)  
**Review independence:** deterministic executor audit; no independent reviewer backend was available

## Checks

- **Ground-truth provenance — PASS:** plant collision, energy, and terminal outcomes come from the synthetic plant; certificates use separate synthetic proof fixtures. No result is labelled real-flight evidence.
- **Score normalization — PASS:** aggregate metrics are arithmetic means and population standard deviations of raw per-run values; no metric is normalized by a model-produced maximum.
- **Result existence — PASS:** 80 `runtime_profile.json` files exist for 4 methods × 4 scenarios × 5 seeds. Every run has config, per-episode, evaluation, and runtime artifacts.
- **Termination completeness — PASS:** each run's termination-reason counts sum to its number of completed episodes.
- **Certified-path software checks — PASS:** all 40 certified runs report mission gate PASS, zero sampled collision episodes, zero uncertified task publications, and zero invalid-κ fallbacks.
- **Scope — WARN:** 10k steps and deterministic synthetic fixtures support pipeline, intervention, and mechanism comparisons only. They do not establish convergence, broad generalization, calibrated physical bounds, or hard WCET.

## Claim Impact

- Supported: the repaired synthetic κ chain is shared by Shield-SAC and Generator-SAC and passes the checked software gate.
- Supported: Generator-SAC reduces measured fallback frequency relative to Shield-SAC in all four fixtures.
- Supported with scope qualifier: no sampled certified-method collision occurred in the tested synthetic runs.
- Unsupported: real-flight safety, unknown-environment safety, global convergence, or monotone improvement across changing certificate epochs.
