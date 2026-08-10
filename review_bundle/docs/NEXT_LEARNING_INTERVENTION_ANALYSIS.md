# Next Learning Intervention Analysis

## Decision boundary

This document is a proposal ledger written while the corrected one-step exposure experiment runs. It does not change that experiment. No intervention below should be implemented until held-out persistent results and fixed-seed visualizations are inspected.

## Current evidence anchors

- Certified support is expressive under the Best-in-Generator oracle.
- The existing 2k policies were each trained on one persistent goal.
- The critic often ranked oracle actions above actor actions, and frozen-critic Q-only updates could exploit that local gradient.
- Counterfactual audits found weak learned goal-conditioned action preference.
- Offline n-step diagnostics strengthened task signal, but prior exposure boundaries also changed Bellman semantics; the corrected one-step experiment removes that confound.
- Physical entropy contains state-dependent support-volume terms, but the normalized-temperature 2k comparison did not improve behavior.

## Competing hypotheses

| Hypothesis | Supporting evidence | Contradicting evidence | Cheapest discriminating experiment | Confounding risk |
|---|---|---|---|---|
| **H1 Temporal-credit problem** | Offline goal-action signal increased from 1 to 3/5/10 steps; one-step soft targets showed weak preference sensitivity. | Immediate reward correctly preferred oracle actions; clean multi-goal one-step training has not yet been tested. | If corrected one-step fails, compare one-step versus exactly `n=3`, same seeds/config. | n-step changes target bias, variance, off-policy error, and completion-boundary handling. |
| **H2 Critic goal-action interaction learning problem** | Learned Q was close to goal offset plus generic action preference; opposite-goal reversals were absent. | The same architecture has not yet received clean broad goal exposure under correct one-step boundaries; finite-width MLP capacity is not proven inadequate. | Supervised fixed-target probe after clean exposure, or one goal-action fusion critic candidate only. | Architecture changes parameter count and optimization geometry. |
| **H3 Actor entropy gradient dominates Q gradient** | Measured entropy/Q parameter-gradient norm ratio was about 3 in the earlier audit. | Q-only actor movement does not prove online failure is solely entropy; normalized-temperature training did not improve 2k results. | Re-audit gradient decomposition on corrected-exposure checkpoints before changing objectives. | Any alpha/entropy change also alters Bellman targets and exploration. |
| **H4 c/G transform worsens optimization conditioning** | `G^T` attenuates physical Q gradients; physical density includes a large state-dependent log-determinant. | Tanh transmission was near one, support oracle succeeded, and certified residual volume was materially nonzero. | Compare normalized-coordinate critic gradients and condition numbers offline on corrected checkpoints. | Critic action normalization and entropy changes can be accidentally conflated. |
| **H5 State-goal correlation / poor local counterfactual coverage** | Historical replay neighborhoods had median one goal direction and no near-opposite goal coverage. | Counterfactual coverage alone did not uniformly restore learned preference in prior frozen analyses. | The running corrected exposure comparison is the direct test; later multiworker collection is a cleaner persistent alternative. | Periodic resets also change state, battery, and lifecycle distributions. |
| **H6 Reward signal weak relative to value scale** | Earlier reward decomposition showed generic costs and soft bootstrap terms much larger than per-step progress. | Immediate oracle-vs-opposite reward contrast was positive in all audited ordinary states; backup penalty semantics were already corrected. | On corrected checkpoints, compare reward contrast, target contrast, and TD error without changing coefficients. | Reward tuning can mask credit/replay problems and changes the task definition. |
| **H7 Action authority still insufficient despite geometric PASS** | Oracle completion was 7/20, not universal; topology and safe support remain conservative. | Oracle made positive progress on 20/20 and center/random controls did not, establishing meaningful policy authority. | Increase oracle evaluation breadth only, not support geometry. | Changing support invalidates the clean learning comparison and safety certificates. |
| **H8 Training horizon simply too short** | Historical failures were 2k; SAC commonly needs more interactions than that. | More steps on one goal do not test random-goal generalization; late windows previously worsened. | Equal-step persistent-only versus corrected exposure at 5k, already running. | Improvement in both groups may be caused by length rather than exposure. |
| **H9 Recurrent/state aliasing problem** | Persistent lifecycle and task age may induce partial observability if the observation omits relevant history. | Current observation includes mission mode, energy, goal/station deltas, and lifecycle features; no direct alias witness exists. | Find pairs with near-identical observations but divergent optimal certified actions/returns. | Adding recurrence changes architecture and optimization simultaneously. |
| **H10 Off-policy replay distribution problem** | Shared replay contains narrow action/goal coverage; actor and target policies evolve; boundary resets alter visitation. | SAC is designed for off-policy replay and critic ranking was often locally useful. | Compare replay goal/action coverage before and after corrected exposure; later test multiworker shared replay. | Replay changes can alter goal coverage, state coverage, and policy lag together. |

## Is n-step the best next intervention?

**Not yet established.** The strongest argument for `n=3` is the offline horizon restoration signal. The strongest argument against implementing it immediately is causal order: historical networks did not receive a genuine multi-goal distribution, and the first exposure implementation also cut bootstrap at collector boundaries. The corrected one-step run must answer whether clean goal diversity alone is enough.

Decision rule after results:

1. If corrected one-step improves held-out task progress/completion and goal sensitivity, continue one-step to the next controlled scale; do not add n-step.
2. If it sees many goals but remains goal-insensitive with poor held-out progress, `n=3` becomes the preferred single-variable candidate because prior offline evidence is already positive.
3. If goal sensitivity improves but control does not, inspect critic target/action ranking and actor exploitation before choosing architecture or entropy changes.
4. If both old persistent and corrected exposure improve at 5k, attribute cautiously to training length and retain the exposure effect estimate relative to the equal-step baseline.

## Recommended future experiment order

1. Inspect corrected exposure 3x5k and fixed seed-100 GIFs.
2. If positive: extend one-step or begin the planned controlled method pilot.
3. If negative: compare one-step versus exactly three-step Generator-SAC, `3 x 2k` or `3 x 5k` as separately authorized.
4. Only after that consider a goal-action interaction critic or strict non-completion relabeling.

No oracle action, demonstration, task waypoint, or relabeled transition should enter training without a new explicit phase.
