# Literature Review: Safe Continuous-Control RL for Persistent UAV Missions

## Scope and evidence policy

This review prioritizes official proceedings, publisher pages, and author manuscripts. “Paper finding” summarizes what the cited work establishes. “Project interpretation” is our scoped comparison to Generator-SAC and is not attributed to the paper. The review does not claim that any cited method solves the present persistent charging problem under the same assumptions.

## 1. Certified / formally safe RL

| Paper | Paper-established fact | Project interpretation |
|---|---|---|
| Alshiekh et al., **Safe Reinforcement Learning via Shielding**, AAAI 2018. [Proceedings](https://doi.org/10.1609/aaai.v32i1.11797) | Synthesizes a reactive shield from temporal-logic specifications and studies conditions under which intervention preserves learner convergence. | A formal shield is a direct baseline category, but its discrete reactive synthesis differs from a continuous certified action set plus recovery atlas. |
| Amani et al., **Safe Reinforcement Learning with Linear Function Approximation**, ICML 2021. [PMLR](https://proceedings.mlr.press/v139/amani21a.html) | Gives no-violation regret guarantees under linear MDP and linear safety-cost assumptions. | Useful for separating strict guarantee claims from empirical deep-control safety; assumptions are not directly shared. |
| Yang et al., **Constrained Update Projection Approach to Safe Policy Optimization**, NeurIPS 2022. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3ba7560b4c3e66d760fbdd472cf4a5a9-Abstract-Conference.html) | Projects policy updates using constrained surrogate bounds. | A policy-update projection baseline tests constrained optimization, not state-level execution certification. |
| Liu et al., **Towards Robust and Safe Reinforcement Learning with Benign Off-policy Data**, ICML 2023. [PMLR](https://proceedings.mlr.press/v202/liu23l.html) | Uses benign off-policy data and a variational policy-improvement stage to improve robustness and safety. | Relevant to replay distribution and safe off-policy learning, but not a substitute for runtime certification. |

## 2. Shielding and backup controllers

| Paper | Paper-established fact | Project interpretation |
|---|---|---|
| Thananjeyan et al., **Recovery RL: Safe Reinforcement Learning with Learned Recovery Zones**, RA-L 2021. [Author manuscript](https://arxiv.org/abs/2010.15920) | Separates a task policy from a learned recovery policy and switches when a safety critic predicts danger. | This is a close algorithmic comparison for task-policy/backup separation. Generator-SAC differs in using certified recovery evidence and a certified continuous task-action set. |
| Bastani and Li, **Safe Reinforcement Learning via Statistical Model Predictive Shielding**, RSS 2021. [RSS proceedings](https://www.roboticsproceedings.org/rss17/p026.html) | Uses a backup policy inside a region checked by statistical model-predictive shielding and proves high-probability safety. | Close to the authority lifecycle and backup-set boundary; guarantee type and uncertainty treatment must be compared explicitly. |
| Kim et al., **Realizable Continuous-Space Shields for Safe Reinforcement Learning**, L4DC 2025. [PMLR](https://proceedings.mlr.press/v283/kim25c.html) | Develops continuous-state/action shields and verifies realizability so a safe corrective action remains available. | Particularly relevant to continuous action correction and nonempty-safe-action requirements. |
| Jansen et al., **Safe Reinforcement Learning via Probabilistic Shields**, 2018. [Author manuscript](https://arxiv.org/abs/1807.06096) | Uses model-checking probabilities to restrict unsafe decisions under uncertainty. | A useful probabilistic-shield baseline, but not equivalent to worst-case continuous post-set containment. |

## 3. Action projection

| Paper | Paper-established fact | Project interpretation |
|---|---|---|
| Chow et al., **Safe Policy Learning for Continuous Control**, CoRL 2020. [PMLR](https://proceedings.mlr.press/v155/chow21a.html) | Uses Lyapunov constraints and projects policy parameters or selected actions into state-dependent feasible sets. | The action-projection variant is a mandatory comparison for whether parameterizing the whole certified support offers optimization or attribution advantages. |
| Cheng et al., **End-to-End Safe Reinforcement Learning through Barrier Functions for Safety-Critical Continuous Control Tasks**, AAAI 2019. [Author manuscript](https://arxiv.org/abs/1903.08792) | Incorporates a differentiable safety layer based on barrier constraints into continuous-control learning. | Closest comparison class for an optimization-based safety layer; certificate scope and recursive recovery differ. |
| Zheng et al., **Safe Reinforcement Learning of Control-Affine Systems with Vertex Networks**, L4DC 2021. [PMLR PDF](https://proceedings.mlr.press/v144/zheng21a/zheng21a.pdf) | Represents safe controls through vertices rather than applying a post-hoc projection. | Conceptually close to learning over a state-dependent feasible action set; compare expressiveness, density semantics, and full-set verification. |

## 4. CBF-based RL

| Paper | Paper-established fact | Project interpretation |
|---|---|---|
| Choi et al., **Reinforcement Learning for Safety-Critical Control under Model Uncertainty, using CLFs and CBFs**, RSS 2020. [RSS proceedings](https://www.roboticsproceedings.org/rss16/p088.html) | Learns model uncertainty terms used inside a CBF-CLF quadratic program and validates on bipedal locomotion. | Strong robotics comparator for combining learning with certified online optimization. |
| Cheng et al., AAAI 2019, above. | Uses barrier-function safety constraints for continuous control. | Compare online QP cost, relative-degree/model assumptions, and recovery-to-charger lifecycle coverage. |

## 5. Reachability / recoverability methods

| Paper | Paper-established fact | Project interpretation |
|---|---|---|
| Yu et al., **Reachability Constrained Reinforcement Learning**, ICML 2022. [PMLR](https://proceedings.mlr.press/v162/yu22d.html) | Represents feasible sets with a safety value function derived from reachability and gives a local convergence result. | Direct conceptual comparator for persistent state constraints and feasible-set learning; our recovery atlas is fixed/certified rather than learned jointly. |
| Qin et al., **Feasible Reachable Policy Iteration**, ICML 2024. [PMLR](https://proceedings.mlr.press/v235/qin24d.html) | Introduces a feasible reachable function coupling goal reachability and safety over a finite horizon. | Important contrast: it explicitly couples goal and safety, whereas the present certificate is task-independent by design. |
| Kokolakis et al., **Reachability Analysis-based Safety-Critical Control using Online Fixed-Time RL**, L4DC 2023. [PMLR](https://proceedings.mlr.press/v211/kokolakis23a.html) | Learns an HJB solution and safe set online in fixed time for a safety-critical control problem. | Relevant to learned reachability versus precomputed recovery certificates. |
| Potteiger et al., **Real-Time Reachability for Neurosymbolic RL-based Safe Autonomous Navigation**, ICoNS 2025. [PMLR](https://proceedings.mlr.press/v288/potteiger25a.html) | Uses real-time reachability to safeguard a goal-conditioned navigation policy on embedded hardware. | Close deployment comparison, though it uses symbolic waypoints that the main Generator-SAC path intentionally excludes. |

## 6. Goal-conditioned RL

| Paper | Paper-established fact | Project interpretation |
|---|---|---|
| Schaul et al., **Universal Value Function Approximators**, ICML 2015. [PMLR](https://proceedings.mlr.press/v37/schaul15.html) | Conditions value functions on goals and demonstrates generalization to unseen goals. | Establishes that goal exposure and representation are first-class variables; one goal per network is not a random-goal learning test. |
| Andrychowicz et al., **Hindsight Experience Replay**, NeurIPS 2017. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2017/hash/453fadbd8a1a3af50a9df4df899537b5-Abstract.html) | Relabels failed off-policy trajectories with achieved goals to improve sparse-goal learning. | Future candidate only. Persistent completion/reassignment semantics require a separate derivation before any relabeling is valid. |
| Naderian et al., **C-Learning: Horizon-Aware Cumulative Accessibility Estimation**, 2020. [Author manuscript](https://arxiv.org/abs/2011.12363) | Learns horizon-conditioned cumulative accessibility for multi-goal reaching. | Supports testing temporal horizon and goal coverage separately rather than assuming either is the unique cause. |

## 7. Multi-step off-policy RL

| Paper | Paper-established fact | Project interpretation |
|---|---|---|
| Munos et al., **Safe and Efficient Off-Policy Reinforcement Learning (Retrace)**, NeurIPS 2016. [Author manuscript](https://arxiv.org/abs/1606.02647) | Uses truncated importance weights for convergent off-policy multi-step evaluation in the tabular setting. | A three-step SAC target is not automatically Retrace; off-policy correction and entropy terms must be derived for the actual replay policy. |
| Daley et al., **Trajectory-Aware Eligibility Traces for Off-Policy RL**, ICML 2023. [PMLR](https://proceedings.mlr.press/v202/daley23a.html) | Studies multi-step off-policy operators and gives tabular convergence conditions for trajectory-aware traces. | A future n-step change is a controlled algorithm modification, not an innocuous logging choice. |
| Schmitt et al., **Off-Policy Actor-Critic with Shared Experience Replay**, ICML 2020. [PMLR](https://proceedings.mlr.press/v119/schmitt20a.html) | Studies actor-critic learning from shared replay and off-policy correction/stability. | Directly informs the proposed multiworker persistent collector: worker diversity changes replay distribution even with one shared learner. |
| Haarnoja et al., **Soft Actor-Critic**, ICML 2018. [PMLR](https://proceedings.mlr.press/v80/haarnoja18b.html) | Defines an off-policy maximum-entropy actor-critic for continuous control. | The physical-density and state-dependent support transform require careful entropy accounting beyond standard fixed action boxes. |

## 8. UAV persistent mission / charging RL

| Paper | Paper-established fact | Project interpretation |
|---|---|---|
| Mathew et al., **Multirobot Rendezvous Planning for Recharging in Persistent Tasks**, T-RO 2015. [IEEE](https://doi.org/10.1109/TRO.2014.2380593) | Plans repeated UAV/charging-robot rendezvous over persistent missions. | Strong non-RL reference for recharge feasibility and mission continuity, but it prespecifies routes/charging points. |
| Mondal et al., **How to Coordinate UAVs and UGVs for Efficient Mission Planning?**, RSS 2025. [RSS proceedings](https://www.roboticsproceedings.org/rss21/p101.html) | Applies DRL to energy-constrained cooperative UAV-UGV routing with recharging. | Relevant task-level comparator, but it is routing/scheduling rather than certified low-level continuous control. |
| Li et al., **Deep RL for Online Routing of UAVs with Wireless Power Transfer**, 2022. [Author manuscript](https://arxiv.org/abs/2204.11477) | Learns combinatorial UAV routes with energy and wireless charging. | Useful charging-aware RL reference; action abstraction and safety guarantees differ substantially. |

## 9. Closest methods to Generator-SAC

The closest comparison is not one paper but four mechanisms:

1. **Recovery/backup switching:** Recovery RL and statistical model-predictive shielding.
2. **Continuous safe-action enforcement:** Lyapunov action projection, CBF safety layers, and continuous-space shields.
3. **Reachability/feasible sets:** RCRL and feasible reachable policy iteration.
4. **Goal-conditioned off-policy learning:** UVFA/HER plus SAC and multi-step off-policy methods.

Generator-SAC’s distinctive empirical object is a single goal-conditioned policy selecting normalized coordinates inside a task-independent, certified, state-dependent zonotope, with a separately certified recovery lifecycle. This description is a project interpretation, not a novelty claim. Novelty requires a systematic closest-work comparison and matched experiments.

## 10. Missing comparisons in the current experiment plan

- **Center-only and random-in-generator:** isolate learned selection from certified support/center effects.
- **Projection-SAC:** same plant, reward, certificate information, and execution budget.
- **Shield/backup SAC:** unconstrained actor plus the same kappa intervention where possible.
- **Penalty/PID-Lagrangian SAC:** empirical constraint optimization without a strict runtime guarantee.
- **Vanilla SAC:** task-learning context, clearly labeled unsafe if it violates constraints.
- **Recovery RL-style learned recovery:** only if guarantee scope is not conflated with certified kappa.
- **Temporal credit:** one-step versus one justified three-step candidate, not a sweep.
- **Goal coverage:** persistent single-worker versus controlled exposure and, later, multiworker persistent collection.
- **Held-out persistent evaluation:** uninterrupted battery, charging, backup, and task stream for every method.
- **Compute/runtime:** certificate construction, online verification, inference, intervention, and training cost.

## Bottom line

The safety literature supports keeping runtime safety evidence separate from task-learning performance. The goal-conditioned and off-policy literature supports treating goal diversity, replay coverage, and return horizon as distinct causal variables. The corrected exposure experiment therefore remains a necessary clean baseline before selecting n-step returns, relabeling, or architectural changes.
