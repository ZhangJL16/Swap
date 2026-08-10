import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SafetyGuideNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return F.softplus(self.head(x)).squeeze(-1)


class MAPPOSafetyGuide:
    """Auxiliary safety plugin for MAPPO.

    It leaves the original actor-critic update untouched and only:
    1. learns a per-agent action risk estimator from warning signals;
    2. revises actions at execution time when the short-term risk exceeds a threshold.
    """

    def __init__(self, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.obs_shape = getattr(args, "raw_obs_shape", args.obs_shape)
        self.device = torch.device(
            f"cuda:{getattr(args, 'gpu_id', 0)}"
            if args.cuda and torch.cuda.is_available()
            else "cpu"
        )
        self.hidden_dim = getattr(args, "safety_hidden_dim", 128)
        self.gamma = self._arg_or_default("safety_gamma", 0.0)
        self.beta = self._arg_or_default("safety_beta", 0.8)
        self.target_update_cycle = int(
            self._arg_or_default("safety_target_update_cycle", 200)
        )
        self.default_threshold = self._arg_or_default("guard_risk_threshold", 0.1)
        self.guard_warmup_steps = int(self._arg_or_default("guard_warmup_steps", 0))
        self.guard_replace_margin = float(
            self._arg_or_default("guard_replace_margin", 0.01)
        )
        self.safety_lr = self._arg_or_default("safety_lr", 3e-4)
        self.current_train_step = 0
        self.last_guard_applied = [0 for _ in range(self.n_agents)]
        self.model_dir = args.model_dir + "/" + args.alg + "/" + args.map

        input_dim = self.obs_shape + self.n_actions
        self.guides = nn.ModuleList(
            [
                SafetyGuideNet(input_dim, self.hidden_dim)
                for _ in range(self.n_agents)
            ]
        ).to(self.device)
        self.target_guides = nn.ModuleList(
            [
                SafetyGuideNet(input_dim, self.hidden_dim)
                for _ in range(self.n_agents)
            ]
        ).to(self.device)
        self.optimizers = [
            torch.optim.Adam(guide.parameters(), lr=self.safety_lr)
            for guide in self.guides
        ]

        for idx in range(self.n_agents):
            self.target_guides[idx].load_state_dict(self.guides[idx].state_dict())

        if getattr(args, "load_model", False):
            self._load_model()

    def _arg_or_default(self, name, default):
        value = getattr(self.args, name, None)
        return default if value is None else value

    def _risk_threshold(self):
        thresholds = {
            "UAV2D": self.default_threshold,
            "UAV3D": self.default_threshold,
            "Basic2P": 20.0,
            "IoV": 50.0,
        }
        return float(thresholds.get(self.args.map, self.default_threshold))

    def _guide_filename(self, agent_idx, prefix=""):
        return os.path.join(self.model_dir, f"{prefix}agent_{agent_idx}_safety_guide.pt")

    def _get_model_paths(self):
        latest_paths = []
        for agent_idx in range(self.n_agents):
            path = self._guide_filename(agent_idx)
            if not os.path.exists(path):
                latest_paths = []
                break
            latest_paths.append(path)
        if latest_paths:
            return latest_paths

        fallback_paths = []
        for agent_idx in range(self.n_agents):
            candidates = glob.glob(self._guide_filename(agent_idx, prefix="*_"))
            if not candidates:
                return None

            def _sort_key(path):
                prefix = os.path.basename(path).split("_", 1)[0]
                return int(prefix) if prefix.isdigit() else -1

            fallback_paths.append(max(candidates, key=_sort_key))
        return fallback_paths

    def _load_model(self):
        model_paths = self._get_model_paths()
        if model_paths is None:
            return
        for agent_idx, path in enumerate(model_paths):
            self.guides[agent_idx].load_state_dict(
                torch.load(path, map_location=self.device)
            )
            self.target_guides[agent_idx].load_state_dict(
                self.guides[agent_idx].state_dict()
            )

    def save_model(self, train_step):
        num = str(train_step // max(getattr(self.args, "save_cycle", 1), 1))
        os.makedirs(self.model_dir, exist_ok=True)
        for agent_idx in range(self.n_agents):
            torch.save(
                self.guides[agent_idx].state_dict(),
                self._guide_filename(agent_idx, prefix=f"{num}_"),
            )
            torch.save(
                self.guides[agent_idx].state_dict(),
                self._guide_filename(agent_idx),
            )

    def _one_hot_actions(self, actions):
        return F.one_hot(actions.long(), num_classes=self.n_actions).float()

    def _build_input(self, obs_tensor, action_tensor):
        return torch.cat([obs_tensor, action_tensor], dim=-1)

    def _predict_action_risk(self, guide, obs_tensor, action_idx):
        action_tensor = self._one_hot_actions(
            torch.tensor([int(action_idx)], device=self.device)
        )
        return guide(self._build_input(obs_tensor, action_tensor)).squeeze(0)

    def _predict_action_risks_batch(self, guide, obs_tensor, action_indices):
        action_indices = torch.as_tensor(
            action_indices, dtype=torch.long, device=self.device
        )
        if action_indices.dim() == 0:
            action_indices = action_indices.unsqueeze(0)
        repeated_obs = obs_tensor.repeat(action_indices.shape[0], 1)
        action_tensor = self._one_hot_actions(action_indices)
        return guide(self._build_input(repeated_obs, action_tensor)).reshape(-1)

    @torch.no_grad()
    def revise_actions(self, observations, avail_actions, base_actions, env):
        self.last_guard_applied = [0 for _ in range(self.n_agents)]
        if self.current_train_step < self.guard_warmup_steps:
            return [int(action) for action in base_actions]
        if not hasattr(env, "estimate_joint_short_risk"):
            return [int(action) for action in base_actions]

        revised_actions = [int(action) for action in base_actions]
        base_short_risks = np.asarray(
            env.estimate_joint_short_risk(revised_actions), dtype=np.float32
        ).reshape(-1)
        collision_flags = None
        if hasattr(env, "estimate_joint_collision_flags"):
            collision_flags = np.asarray(
                env.estimate_joint_collision_flags(revised_actions), dtype=np.float32
            ).reshape(-1)

        for agent_idx in range(self.n_agents):
            base_action = int(revised_actions[agent_idx])
            base_short = float(base_short_risks[agent_idx])
            if collision_flags is not None:
                if float(collision_flags[agent_idx]) <= 0.0:
                    continue
            else:
                threshold = self._risk_threshold()
                if base_short < threshold:
                    continue

            obs_tensor = torch.tensor(
                observations[agent_idx], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            candidate_action_indices = [
                int(action_idx)
                for action_idx in range(self.n_actions)
                if avail_actions[agent_idx] is None
                or float(avail_actions[agent_idx][action_idx]) > 0
            ]
            if not candidate_action_indices:
                continue

            candidate_long_values = (
                self._predict_action_risks_batch(
                    self.guides[agent_idx], obs_tensor, candidate_action_indices
                )
                .detach()
                .cpu()
                .numpy()
            )
            candidate_long_map = {
                action_idx: float(long_value)
                for action_idx, long_value in zip(
                    candidate_action_indices, candidate_long_values
                )
            }

            base_long = candidate_long_map.get(base_action)
            if base_long is None:
                base_long = float(
                    self._predict_action_risk(
                        self.guides[agent_idx], obs_tensor, base_action
                    ).item()
                )
            base_total = base_short + self.beta * base_long

            best_action = base_action
            best_total = base_total
            candidate_joint_actions = []
            for action_idx in candidate_action_indices:
                candidate_actions = list(revised_actions)
                candidate_actions[agent_idx] = int(action_idx)
                candidate_joint_actions.append(candidate_actions)

            if hasattr(env, "estimate_joint_short_risk_batch"):
                candidate_short_all = np.asarray(
                    env.estimate_joint_short_risk_batch(candidate_joint_actions),
                    dtype=np.float32,
                )[:, agent_idx]
            else:
                candidate_short_all = np.asarray(
                    [
                        env.estimate_joint_short_risk(candidate_actions)[agent_idx]
                        for candidate_actions in candidate_joint_actions
                    ],
                    dtype=np.float32,
                )

            for offset, action_idx in enumerate(candidate_action_indices):
                candidate_short = float(candidate_short_all[offset])
                candidate_long = candidate_long_map[action_idx]
                candidate_total = candidate_short + self.beta * candidate_long
                if candidate_total < best_total:
                    best_total = candidate_total
                    best_action = int(action_idx)

            if (
                best_action != base_action
                and base_total > best_total + self.guard_replace_margin
            ):
                revised_actions[agent_idx] = best_action
                self.last_guard_applied[agent_idx] = 1

        return revised_actions

    def learn(self, batch, max_episode_len, train_step):
        self.current_train_step = int(train_step)
        def _to_tensor(value, dtype):
            if isinstance(value, torch.Tensor):
                return value.to(device=self.device, dtype=dtype)
            return torch.as_tensor(value, dtype=dtype, device=self.device)

        obs_source = batch.get("o_raw", batch["o"])
        next_obs_source = batch.get("o_next_raw", batch["o_next"])
        obs = _to_tensor(obs_source, torch.float32)
        next_obs = _to_tensor(next_obs_source, torch.float32)
        actions = _to_tensor(batch["u"], torch.long).squeeze(-1)
        avail_actions = _to_tensor(batch["avail_u"], torch.float32)
        avail_actions_next = _to_tensor(batch["avail_u_next"], torch.float32)
        terminated = _to_tensor(batch["terminated"], torch.float32).squeeze(-1)
        mask = 1 - _to_tensor(batch["padded"], torch.float32).squeeze(-1)
        active_mask = None
        if "agent_active_mask" in batch:
            active_mask = _to_tensor(batch["agent_active_mask"], torch.float32).squeeze(-1)
        warning = _to_tensor(batch["warning_signal"], torch.float32).squeeze(-1)
        losses = []

        for agent_idx in range(self.n_agents):
            agent_obs = obs[:, :, agent_idx, :]
            agent_next_obs = next_obs[:, :, agent_idx, :]
            agent_actions = actions[:, :, agent_idx]
            agent_avail_next = avail_actions_next[:, :, agent_idx, :]
            action_one_hot = self._one_hot_actions(agent_actions)

            guide_input = self._build_input(agent_obs, action_one_hot)
            pred_taken = self.guides[agent_idx](guide_input)

            with torch.no_grad():
                next_action_risks = []
                for action_idx in range(self.n_actions):
                    candidate_actions = torch.full_like(
                        agent_actions, int(action_idx), device=self.device
                    )
                    candidate_one_hot = self._one_hot_actions(candidate_actions)
                    candidate_input = self._build_input(agent_next_obs, candidate_one_hot)
                    candidate_risk = self.target_guides[agent_idx](candidate_input)
                    if agent_avail_next.numel() > 0:
                        invalid_mask = agent_avail_next[:, :, action_idx] <= 0
                        candidate_risk = candidate_risk.masked_fill(
                            invalid_mask, float("inf")
                        )
                    next_action_risks.append(candidate_risk.unsqueeze(-1))
                next_pred = torch.cat(next_action_risks, dim=-1)
                min_next = torch.min(next_pred, dim=-1)[0]
                min_next = torch.where(
                    torch.isfinite(min_next),
                    min_next,
                    torch.zeros_like(min_next),
                )
                # The guide learns long-term risk only; the current-step
                # immediate risk is kept in the environment short-risk signal
                # and is combined with the guide output only at action revision.
                target = self.gamma * min_next * (1 - terminated)

            agent_mask = mask
            if active_mask is not None:
                agent_mask = agent_mask * active_mask[:, :, agent_idx]
            td_error = (pred_taken - target) * agent_mask
            denom = agent_mask.sum().clamp(min=1.0)
            loss = (td_error.pow(2)).sum() / denom

            self.optimizers[agent_idx].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.guides[agent_idx].parameters(),
                getattr(self.args, "grad_norm_clip", 10),
            )
            self.optimizers[agent_idx].step()
            losses.append(float(loss.item()))

        if train_step > 0 and train_step % self.target_update_cycle == 0:
            for agent_idx in range(self.n_agents):
                self.target_guides[agent_idx].load_state_dict(
                    self.guides[agent_idx].state_dict()
                )
        return float(sum(losses) / len(losses)) if losses else None
