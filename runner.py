from math import e
import matplotlib
matplotlib.use("Agg")
from matplotlib.style import available
import csv
import numpy as np
import pandas as pd
import os
from agent import agent
from common.rollout import RolloutWorker, CommRolloutWorker, SMAC_MAPS
from common.seeding import (
    derive_episode_seed,
    preserve_rng_state,
    reset_env_with_seed,
    temporary_seed,
)
from common.parallel_smac_rollout import ParallelSMACEpisodeCollector
from agent.agent import Agents, CommAgents
from common.replay_buffer import ReplayBuffer
from common.replay_buffer_RGMComm import Buffer
import matplotlib.pyplot as plt
import scipy.io as sio
from tqdm import trange, tqdm 
import torch
from datetime import datetime

from safety.constrained_flow import ConstrainedFlow
from safety.correction_buffer import CorrectionBuffer
from safety.distillation_trainer import DistillationTrainer
from safety.distilled_corrector import DistilledCorrector
from safety.flow_trainer import FlowTrainer
from hierarchy.energy_buffer import EnergyReplayBuffer
from hierarchy.navigation_energy import NavigationEnergyNetwork, NavigationEnergyTrainer
from hierarchy.order_q import OrderQNetwork, OrderQTrainer
from hierarchy.order_replay_buffer import OrderReplayBuffer


UAV_COLLISION_MAPS = {
    "UAV3D",
    "UAVEncircle",
    "UAVencircle",
    "UAVDelivery",
    "UAVDelivery2D",
    "UAVDelivery3D",
    "UAVEnergyDelivery",
    "UAVEnergyDelivery2D",
    "UAVEnergyDelivery3D",
    "UAVEnergyDeliveryLevel",
    "UAVEnergyDeliveryLevel2D",
    "UAVEnergyDeliveryLevel3D",
    "UAVEnergyDeliveryHierarchical",
}

UAV_DELIVERY_MAPS = {
    "UAVDelivery",
    "UAVDelivery2D",
    "UAVDelivery3D",
    "UAVEnergyDelivery",
    "UAVEnergyDelivery2D",
    "UAVEnergyDelivery3D",
    "UAVEnergyDeliveryLevel",
    "UAVEnergyDeliveryLevel2D",
    "UAVEnergyDeliveryLevel3D",
    "UAVEnergyDeliveryHierarchical",
}

DEFAULT_CSV_COLUMNS = [
    "event",
    "episode",
    "timestep",
    "train_step",
    "progress",
    "episode_idx",
    "episode_reward",
    "episode_steps",
    "epsilon",
    "win_rate",
    "collision_count",
    "obstacle_collision_count",
    "agent_collision_count",
    "safety_loss",
    "agent_health",
    "enemy_health",
    "agent_alive",
    "eval_step",
    "timestamp",
]

UAV_DELIVERY_CSV_COLUMNS = [
    "event",
    "episode",
    "timestep",
    "train_step",
    "progress",
    "episode_idx",
    "episode_reward",
    "episode_steps",
    "epsilon",
    "delivery_success_rate",
    "orders_completed",
    "total_orders",
    "active_orders",
    "available_orders",
    "picked_orders",
    "idle_agents",
    "powered_agents",
    "depleted_agents",
    "mean_energy",
    "charging_agents",
    "mean_goal_distance",
    "healthy_agents",
    "collision_count",
    "obstacle_collision_count",
    "agent_collision_count",
    "safety_loss",
    "eval_episode_steps",
    "timestamp",
]

UAV_DELIVERY_SUMMARY_KEYS = [
    "orders_completed",
    "total_orders",
    "active_orders",
    "available_orders",
    "picked_orders",
    "idle_agents",
    "powered_agents",
    "depleted_agents",
    "mean_energy",
    "charging_agents",
    "mean_goal_distance",
    "healthy_agents",
]

UAV_DELIVERY_DIAGNOSTIC_KEYS = [
    "high_decision_count",
    "sampled_charge_rate",
    "sampled_order_rate",
    "executed_charge_rate",
    "executed_order_rate",
    "executed_idle_rate",
    "order_locked_rate",
    "charge_locked_rate",
    "mode_train_mask_mean",
    "auction_calls",
    "auction_candidate_mean",
    "auction_order_mean",
    "auction_assignment_mean",
    "order_commit_attempt_rate",
    "order_commit_success_rate",
    "charge_fallback_rate",
    "order_exec_steps",
    "order_pickup_steps",
    "order_delivery_steps",
    "order_pickup_success_count",
    "order_delivery_success_count",
    "order_progress_mean",
    "order_positive_progress_rate",
    "order_regress_rate",
    "order_target_distance_mean",
    "pickup_success_per_pickup_step",
    "delivery_success_per_delivery_step",
    "charge_exec_steps",
    "idle_exec_steps",
    "order_obstacle_collision_count",
    "order_agent_collision_count",
    "charge_obstacle_collision_count",
    "charge_agent_collision_count",
    "idle_obstacle_collision_count",
    "idle_agent_collision_count",
    "order_collision_rate",
    "charge_collision_rate",
    "idle_collision_rate",
]

UAV_DELIVERY_CSV_COLUMNS = (
    UAV_DELIVERY_CSV_COLUMNS[:-1]
    + UAV_DELIVERY_DIAGNOSTIC_KEYS
    + UAV_DELIVERY_CSV_COLUMNS[-1:]
)

UAV_DELIVERY_EXPERIMENT_LOG = os.path.join(
    "train_logs", "uav_delivery_experiments.csv"
)

UAV_DELIVERY_EXPERIMENT_COLUMNS = [
    "device",
    "rl_algorithm",
    "map",
    "timestamp",
    "n_agents",
    "configured_total_orders",
    "max_active_orders",
    "n_steps",
    "final_timestep",
    "train_step",
    "eval_episode_reward",
    "delivery_success_rate",
    "orders_completed",
    "total_orders",
    "collision_count",
    "obstacle_collision_count",
    "agent_collision_count",
    "eval_episode_steps",
    "run_script",
    "run_command",
    "seed",
    "eval_seed",
    "hmappo_meta_period",
    "high_level_n_actions",
    "charge_action_id",
    "high_lr_actor",
    "high_lr_critic",
    "high_actor_hidden_dim",
    "high_critic_hidden_dim",
]


class Runner:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.parallel_episode_collector = None

        if (
            args.alg.find("commnet") > -1 or args.alg.find("g2anet") > -1
        ):  # communication agent
            self.agents = CommAgents(args)
            self.rolloutWorker = CommRolloutWorker(env, self.agents, args)
        elif args.alg.lower().find("rgmcomm") > -1:
            
            self.args.obs_shape = [args.obs_shape for _ in range(args.n_agents)]
            self.args.action_shape = [args.n_actions for _ in range(args.n_agents)]
            self.noise = self.args.noise_rate
            self.epsilon = self.args.epsilon

            self.agents = self._init_agents()
            self.buffer = Buffer(args)
            self.rolloutWorker = RolloutWorker(env, self.agents, args)
            
        else:  # no communication agent
            self.agents = Agents(args, env)
            self.rolloutWorker = RolloutWorker(env, self.agents, args)
        if (
            args.map in SMAC_MAPS
            and int(getattr(args, "smac_parallel_envs", 1)) > 1
            and (args.alg.find("qmix") != -1 or args.alg.find("vdn") != -1)
        ):
            self.parallel_episode_collector = ParallelSMACEpisodeCollector(
                args, self.agents
            )
        if (
            not args.evaluate
            and args.alg.find("coma") == -1
            and args.alg.find("central_v") == -1
            and args.alg.find("reinforce") == -1
            and args.alg.find("ippo") == -1
            and args.alg.find("mappo") == -1
            and args.alg.find("macpo") == -1
            and args.alg.lower().find("rgmcomm") == -1
        ):  # these algorithms are on-policy
            self.buffer = ReplayBuffer(args)
        self.args = args
        self.win_rates = []
        self.episode_rewards = []

        self.warning_signals = []

        self.smac_summary = {
            "win_rate": [],
            "episode_reward": [],
            "collision_count": [],
            "obstacle_collision_count": [],
            "agent_collision_count": [],
            "step": [],
            "agent_health": [],
            "enemy_health": [],
            "agent_alive": [],
        }
        if args.map in UAV_DELIVERY_MAPS:
            self.smac_summary = {
                "win_rate": [],
                "episode_reward": [],
                "collision_count": [],
                "obstacle_collision_count": [],
                "agent_collision_count": [],
                "step": [],
                "orders_completed": [],
                "total_orders": [],
                "active_orders": [],
                "available_orders": [],
                "picked_orders": [],
                "idle_agents": [],
                "powered_agents": [],
                "depleted_agents": [],
                "mean_energy": [],
                "charging_agents": [],
                "mean_goal_distance": [],
                "healthy_agents": [],
            }

        self.maze_summary = {
            "reward": [],
            "bonus": [],
            "trap": [],
            "life": [],
            "step": [],
            "arrived": [],
            "died": [],
            "warning_signal": [],
        }
        for i in range(self.env.n_agents):
            self.maze_summary.update({f"agent{i}_trap": [], f"agent{i}_life": []})

        self.IoV_summary = {
            "reward": [],
            "latency": [],
            "protection_level": [],
            "atk_succ_rate": [],
            "malicious_msg_nums": [],
        }

        self.stego_summary = {
            "psnr": [],
            "ssim": [],
            "uiqi": [],
            "accuracy": [],
            "rsbpp": [],
            "error": [],
            "mse": [],
            "density": [],
            "latency": [],
            "reward": [],
        }

        self.summary_agent = {f'{agent}': {} for agent in range(self.env.n_agents)}
        self.add_metrics()

        # 用来保存plt和pkl
        # 统一保存路径: result/地图/算法/
        self.save_path = self.args.result_dir + "/" + args.map + "/" + args.alg + "/"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.csv_columns = (
            UAV_DELIVERY_CSV_COLUMNS
            if self.args.map in UAV_DELIVERY_MAPS
            else DEFAULT_CSV_COLUMNS
        )
        self.log_path = self._init_csv_log()
        self.train_episode_count = 0
        self.rgm_train_episode_index = 0
        self.eval_count = 0
        self.last_eval_summary = None
        self.last_eval_time_steps = None
        self.last_eval_train_steps = None
        self._last_saved_bucket = -1
        if self.args.map == "UAV3D":
            self.args.save_cycle = 50000
        self._init_cbf_flow_phase_modules()

    def _base_env(self):
        return self.env.env if hasattr(self.env, "env") else self.env

    def _init_cbf_flow_phase_modules(self):
        self.correction_buffer = None
        self.flow_net = None
        self.flow_trainer = None
        self.student = None
        self.student_trainer = None
        self.energy_buffer = None
        self.energy_model = None
        self.energy_trainer = None
        self.order_buffer = None
        self.order_q_net = None
        self.order_q_trainer = None
        if self.args.alg != "hmappo_cbf_flow":
            return
        self.cbf_flow_artifact_dir = (
            getattr(self.args, "cbf_flow_artifact_dir", "")
            or os.path.join(self.args.model_dir, self.args.alg, self.args.map, "cbf_flow")
        )
        self.cbf_flow_load_artifact_dir = (
            getattr(self.args, "cbf_flow_load_artifact_dir", "")
            or self.cbf_flow_artifact_dir
        )
        base_env = self._base_env()
        safety_state_dim = int(
            len(base_env.get_cbf_safety_state())
            if hasattr(base_env, "get_cbf_safety_state")
            else self.args.n_agents * int(getattr(self.args, "low_action_dim", self.args.n_actions)) * 2
        )
        joint_action_dim = int(
            self.args.n_agents * int(getattr(self.args, "low_action_dim", self.args.n_actions))
        )
        self.correction_buffer = CorrectionBuffer(getattr(self.args, "energy_buffer_size", 100000))
        self.flow_net = ConstrainedFlow(
            safety_state_dim=safety_state_dim,
            joint_action_dim=joint_action_dim,
            hidden_dim=getattr(self.args, "flow_hidden_dim", 256),
        )
        self.flow_trainer = FlowTrainer(
            self.flow_net,
            lr=getattr(self.args, "flow_lr", 3e-4),
            flow_train_steps=getattr(self.args, "flow_train_steps", 8),
            cbf_coef=getattr(self.args, "flow_cbf_coef", 1.0),
            endpoint_coef=getattr(self.args, "flow_endpoint_coef", 1.0),
            terminal_coef=getattr(self.args, "flow_terminal_coef", 1.0),
            flow_margin=getattr(self.args, "flow_margin", 0.0),
            kappa=getattr(self.args, "flow_kappa", 1.0),
            device=(
                f"cuda:{getattr(self.args, 'gpu_id', 0)}"
                if getattr(self.args, "cuda", False) and torch.cuda.is_available()
                else "cpu"
            ),
        )
        self.student = DistilledCorrector(
            safety_state_dim=safety_state_dim,
            joint_action_dim=joint_action_dim,
            hidden_dim=getattr(self.args, "flow_hidden_dim", 256),
        )
        self.student_trainer = DistillationTrainer(
            self.student,
            lr=getattr(self.args, "student_lr", 3e-4),
            teacher_coef=getattr(self.args, "student_teacher_coef", 1.0),
            safety_coef=getattr(self.args, "student_safety_coef", 1.0),
            margin=getattr(self.args, "student_margin", 0.0),
            device=(
                f"cuda:{getattr(self.args, 'gpu_id', 0)}"
                if getattr(self.args, "cuda", False) and torch.cuda.is_available()
                else "cpu"
            ),
        )
        self.energy_buffer = EnergyReplayBuffer(getattr(self.args, "energy_buffer_size", 100000))
        self.energy_model = NavigationEnergyNetwork()
        self.energy_trainer = NavigationEnergyTrainer(
            self.energy_model,
            lr=getattr(self.args, "energy_lr", 3e-4),
            gamma=getattr(self.args, "energy_gamma", 0.99),
            device=(
                f"cuda:{getattr(self.args, 'gpu_id', 0)}"
                if getattr(self.args, "cuda", False) and torch.cuda.is_available()
                else "cpu"
            ),
        )
        self.order_buffer = OrderReplayBuffer(getattr(self.args, "order_q_buffer_size", 100000))
        self.order_q_net = OrderQNetwork()
        self.order_q_trainer = OrderQTrainer(
            self.order_q_net,
            lr=getattr(self.args, "order_q_lr", 3e-4),
            gamma=getattr(self.args, "order_q_gamma", 0.99),
            device=(
                f"cuda:{getattr(self.args, 'gpu_id', 0)}"
                if getattr(self.args, "cuda", False) and torch.cuda.is_available()
                else "cpu"
            ),
        )
        if hasattr(base_env, "high_controller"):
            base_env.high_controller.order_q_net = self.order_q_net
            base_env.high_controller.energy_model = self.energy_model
        self._load_cbf_flow_artifacts()

    def _artifact_path(self, name, root=None):
        root = self.cbf_flow_artifact_dir if root is None else root
        return os.path.join(root, name)

    def _load_cbf_flow_artifacts(self):
        if self.args.alg != "hmappo_cbf_flow":
            return
        root = getattr(self, "cbf_flow_load_artifact_dir", "")
        if not root or not os.path.isdir(root):
            return

        def _load_state(module, name):
            path = self._artifact_path(name, root=root)
            if module is not None and os.path.exists(path):
                module.load_state_dict(torch.load(path, map_location="cpu"))

        _load_state(self.flow_net, "flow.pt")
        _load_state(self.student, "student.pt")
        _load_state(self.energy_model, "navigation_energy.pt")
        _load_state(self.order_q_net, "order_q.pt")
        if self.flow_net is not None and self.flow_trainer is not None:
            self.flow_trainer.flow_net = self.flow_net.to(self.flow_trainer.device)
        if self.student is not None and self.student_trainer is not None:
            self.student_trainer.student = self.student.to(self.student_trainer.device)
        if self.energy_model is not None and self.energy_trainer is not None:
            self.energy_trainer.model = self.energy_model.to(self.energy_trainer.device)
            self.energy_trainer.target.load_state_dict(self.energy_trainer.model.state_dict())
        if self.order_q_net is not None and self.order_q_trainer is not None:
            self.order_q_trainer.q_net = self.order_q_net.to(self.order_q_trainer.device)
            self.order_q_trainer.target_q_net.load_state_dict(
                self.order_q_trainer.q_net.state_dict()
            )

        if self.correction_buffer is not None:
            path = self._artifact_path("correction_buffer.pkl", root=root)
            if os.path.exists(path):
                self.correction_buffer.load(path)
        if self.energy_buffer is not None:
            path = self._artifact_path("energy_buffer.pkl", root=root)
            if os.path.exists(path):
                self.energy_buffer.load(path)
        if self.order_buffer is not None:
            path = self._artifact_path("order_buffer.pkl", root=root)
            if os.path.exists(path):
                self.order_buffer.load(path)

    def _save_cbf_flow_artifacts(self, train_step=None, final=False):
        if self.args.alg != "hmappo_cbf_flow":
            return
        os.makedirs(self.cbf_flow_artifact_dir, exist_ok=True)

        def _save_state(module, name):
            if module is None:
                return
            torch.save(module.state_dict(), self._artifact_path(name))
            if train_step is not None:
                torch.save(module.state_dict(), self._artifact_path(f"{train_step}_{name}"))

        _save_state(self.flow_net, "flow.pt")
        _save_state(self.student, "student.pt")
        _save_state(self.energy_model, "navigation_energy.pt")
        _save_state(self.order_q_net, "order_q.pt")
        if self.correction_buffer is not None:
            self.correction_buffer.save(self._artifact_path("correction_buffer.pkl"))
        if self.energy_buffer is not None:
            self.energy_buffer.save(self._artifact_path("energy_buffer.pkl"))
        if self.order_buffer is not None:
            self.order_buffer.save(self._artifact_path("order_buffer.pkl"))
        if final and bool(getattr(self.args, "cbf_flow_export_student", False)):
            metrics = self._validate_student_for_export()
            if metrics.get("student_cbf_violation_rate", 1.0) == 0.0 and metrics.get(
                "student_action_mse", float("inf")
            ) <= float(getattr(self.args, "cbf_flow_student_mse_threshold", 0.02)):
                torch.save(self.student.state_dict(), self._artifact_path("student_deploy.pt"))

    def _validate_student_for_export(self):
        if (
            self.student is None
            or self.correction_buffer is None
            or len(self.correction_buffer) < 1
        ):
            return {"student_cbf_violation_rate": 1.0, "student_action_mse": float("inf")}
        batch = self.correction_buffer.sample(
            min(int(getattr(self.args, "energy_batch_size", 256)), len(self.correction_buffer))
        )
        device = next(self.student.parameters()).device
        with torch.no_grad():
            state = torch.as_tensor(np.stack(batch["safety_state_abs"]), dtype=torch.float32, device=device)
            raw = torch.as_tensor(np.stack(batch["raw_joint_action"]), dtype=torch.float32, device=device)
            correct = torch.as_tensor(np.stack(batch["correct_joint_action"]), dtype=torch.float32, device=device)
            action = self.student(state, raw)
            mse = torch.mean((action - correct).pow(2)).item()
            A, c = self.student_trainer._constraint_tensors(batch)
            margin = torch.bmm(A, action.unsqueeze(-1)).squeeze(-1) + c
            violation_rate = float((margin < float(getattr(self.args, "student_margin", 0.0))).float().mean().item())
        return {
            "student_action_mse": float(mse),
            "student_cbf_violation_rate": float(violation_rate),
        }

    def _drain_correction_records(self):
        if self.correction_buffer is None or not hasattr(self.agents, "drain_correction_records"):
            return 0
        records = self.agents.drain_correction_records()
        for record in records:
            self.correction_buffer.add(record)
        return len(records)

    def _drain_order_records(self):
        if self.order_buffer is None:
            return 0
        base_env = self._base_env()
        if not hasattr(base_env, "get_completed_option_transitions"):
            return 0
        records = base_env.get_completed_option_transitions()
        for record in records:
            self.order_buffer.add(record)
        return len(records)

    def _drain_energy_records(self):
        if self.energy_buffer is None or not hasattr(self.agents, "drain_energy_records"):
            return 0
        records = self.agents.drain_energy_records()
        for record in records:
            self.energy_buffer.add(record)
        return len(records)

    def _train_cbf_flow_phase(self):
        phase = getattr(self.args, "training_phase", "low_cbf")
        if self.args.alg != "hmappo_cbf_flow" or self.correction_buffer is None:
            return {}
        if phase == "low_cbf":
            return {}
        if phase == "energy":
            if self.energy_buffer is None or self.energy_trainer is None or len(self.energy_buffer) < 1:
                return {"energy_buffer_size": float(len(self.energy_buffer or []))}
            batch = self.energy_buffer.sample(
                min(int(getattr(self.args, "energy_batch_size", 64)), len(self.energy_buffer))
            )
            return self.energy_trainer.train_step(batch)
        if phase == "high_q":
            if self.order_buffer is None or self.order_q_trainer is None or len(self.order_buffer) < 1:
                return {"order_q_buffer_size": float(len(self.order_buffer or []))}
            batch = self.order_buffer.sample(
                min(int(getattr(self.args, "order_q_batch_size", 64)), len(self.order_buffer))
            )
            return self.order_q_trainer.train_step(batch)
        if len(self.correction_buffer) < 1:
            return {"correction_buffer_size": 0.0}
        batch_size = min(
            int(getattr(self.args, "energy_batch_size", 64)),
            len(self.correction_buffer),
        )
        batch = self.correction_buffer.sample(batch_size)
        metrics = {"correction_buffer_size": float(len(self.correction_buffer))}
        if phase == "flow" and self.flow_trainer is not None:
            metrics.update(self.flow_trainer.train_step(batch))
        elif phase == "distill" and self.flow_net is not None and self.student_trainer is not None:
            device = self.flow_trainer.device if self.flow_trainer is not None else torch.device("cpu")
            with torch.no_grad():
                state = torch.as_tensor(
                    np.stack(batch["safety_state_abs"]),
                    dtype=torch.float32,
                    device=device,
                )
                raw = torch.as_tensor(
                    np.stack(batch["raw_joint_action"]),
                    dtype=torch.float32,
                    device=device,
                )
                flow_action = self.flow_net.integrate(
                    state,
                    raw,
                    int(getattr(self.args, "flow_eval_steps", getattr(self.args, "flow_train_steps", 8))),
                )
            batch["flow_joint_action"] = flow_action.detach().cpu().numpy()
            metrics.update(self.student_trainer.train_step(batch))
        return metrics

    def close(self):
        if self.parallel_episode_collector is not None:
            self.parallel_episode_collector.close()
            self.parallel_episode_collector = None

    def _get_csv_log_path(self):
        log_dir = os.path.join("train_logs", self.args.alg, self.args.map)
        os.makedirs(log_dir, exist_ok=True)
        log_idx = 0
        while True:
            log_path = os.path.join(log_dir, f"{self.args.map}_log_{log_idx}.csv")
            if not os.path.exists(log_path):
                return log_path
            log_idx += 1

    def _init_csv_log(self):
        log_path = self._get_csv_log_path()
        with open(log_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(self.csv_columns)
        return log_path

    def _append_log_row(self, row):
        with open(self.log_path, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(row)

    def _uav_delivery_experiment_log_path(self):
        return (
            getattr(self.args, "experiment_log_csv", "")
            or os.environ.get("MARL_EXPERIMENT_LOG_CSV", "")
            or UAV_DELIVERY_EXPERIMENT_LOG
        )

    def _normalize_uav_delivery_experiment_log(self):
        log_path = self._uav_delivery_experiment_log_path()
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        if (
            not os.path.exists(log_path)
            or os.path.getsize(log_path) == 0
        ):
            return

        with open(log_path, newline="", encoding="utf-8") as csv_file:
            rows = list(csv.reader(csv_file))
        if not rows:
            return

        header = rows[0]
        if header == UAV_DELIVERY_EXPERIMENT_COLUMNS:
            return

        overflow_columns = [
            column
            for column in UAV_DELIVERY_EXPERIMENT_COLUMNS
            if column not in header
        ]
        normalized_rows = [UAV_DELIVERY_EXPERIMENT_COLUMNS]
        for row in rows[1:]:
            values = {
                column: row[idx] if idx < len(row) else ""
                for idx, column in enumerate(header)
            }
            if len(row) > len(header):
                for idx, value in enumerate(row[len(header):]):
                    if idx < len(overflow_columns):
                        values[overflow_columns[idx]] = value
            normalized_rows.append(
                [values.get(column, "") for column in UAV_DELIVERY_EXPERIMENT_COLUMNS]
            )

        with open(log_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(normalized_rows)

    def _write_uav_delivery_experiment_row(self, row):
        log_path = self._uav_delivery_experiment_log_path()
        self._normalize_uav_delivery_experiment_log()
        write_header = (
            not os.path.exists(log_path)
            or os.path.getsize(log_path) == 0
        )
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            if write_header:
                writer.writerow(UAV_DELIVERY_EXPERIMENT_COLUMNS)
            writer.writerow(row)

    def _uav_delivery_experiment_level_fields(self):
        optional_values = [
            getattr(self.args, "high_lr_actor", ""),
            getattr(self.args, "high_lr_critic", ""),
            getattr(self.args, "high_actor_hidden_dim", ""),
            getattr(self.args, "high_critic_hidden_dim", ""),
        ]
        return [
            int(getattr(self.args, "seed", 0)),
            int(getattr(self.args, "eval_seed", 0)),
            int(getattr(self.args, "hmappo_meta_period", 0)),
            int(getattr(self.args, "high_level_n_actions", 0)),
            int(getattr(self.args, "charge_action_id", -1)),
            *["" if value is None else value for value in optional_values],
        ]

    def _append_uav_delivery_experiment_result(self, time_steps, train_steps):
        if self.args.map not in UAV_DELIVERY_MAPS or self.last_eval_summary is None:
            return

        summary = self.last_eval_summary
        row = [
            getattr(self.args, "experiment_device", "dorm"),
            self.args.alg,
            self.args.map,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            int(getattr(self.args, "n_agents", 0)),
            int(getattr(self.args, "uav_total_orders", 0)),
            int(getattr(self.args, "uav_max_active_orders", 0)),
            int(getattr(self.args, "n_steps", 0)),
            int(time_steps if time_steps is not None else 0),
            int(train_steps if train_steps is not None else 0),
            float(summary.get("episode_reward", 0.0)),
            float(summary.get("win_rate", 0.0)),
            float(summary.get("orders_completed", 0.0)),
            float(summary.get("total_orders", 0.0)),
            float(summary.get("collision_count", 0.0)),
            float(summary.get("obstacle_collision_count", 0.0)),
            float(summary.get("agent_collision_count", 0.0)),
            float(summary.get("step", 0.0)),
            getattr(self.args, "run_script", ""),
            getattr(self.args, "run_command", ""),
        ]
        row.extend(self._uav_delivery_experiment_level_fields())
        self._write_uav_delivery_experiment_row(row)

    def _append_uav_delivery_rgm_experiment_result(
        self, result, result_episode, time_steps
    ):
        if self.args.map not in UAV_DELIVERY_MAPS:
            return

        summary = self.env.summary() if hasattr(self.env, "summary") else {}
        episode_rewards = result_episode.get("reward", [])
        if episode_rewards:
            episode_reward = float(np.sum(episode_rewards))
        else:
            completed_rewards = result.get("reward", [])
            episode_reward = float(completed_rewards[-1]) if completed_rewards else 0.0
        win_history = result.get("win_rate", [])
        win_rate = (
            float(win_history[-1])
            if win_history
            else float(bool(summary.get("win_tag", False)))
        )
        row = [
            getattr(self.args, "experiment_device", "dorm"),
            self.args.alg,
            self.args.map,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            int(getattr(self.args, "n_agents", 0)),
            int(getattr(self.args, "uav_total_orders", 0)),
            int(getattr(self.args, "uav_max_active_orders", 0)),
            int(getattr(self.args, "time_steps", getattr(self.args, "n_steps", 0))),
            int(time_steps if time_steps is not None else 0),
            int(time_steps if time_steps is not None else 0),
            episode_reward,
            win_rate,
            float(summary.get("orders_completed", 0.0)),
            float(summary.get("total_orders", 0.0)),
            float(summary.get("collision_count", 0.0)),
            float(summary.get("obstacle_collision_count", 0.0)),
            float(summary.get("agent_collision_count", 0.0)),
            float(summary.get("step", 0.0)),
            getattr(self.args, "run_script", ""),
            getattr(self.args, "run_command", ""),
        ]
        row.extend(self._uav_delivery_experiment_level_fields())
        self._write_uav_delivery_experiment_row(row)

    def _set_log_row_value(self, row, column, value):
        try:
            row[self.csv_columns.index(column)] = value
        except ValueError:
            return

    def _summary_float(self, summary, key, default=0.0):
        return float(summary.get(key, default))

    def _reset_rgm_env_for_episode(self, evaluate=False, episode_index=None):
        seed = derive_episode_seed(
            self.args,
            evaluate=evaluate,
            episode_index=episode_index
            if evaluate
            else self.rgm_train_episode_index,
        )
        if evaluate:
            return reset_env_with_seed(self.env, seed)

        with preserve_rng_state(include_torch=False):
            result = reset_env_with_seed(self.env, seed)
        self.rgm_train_episode_index += 1
        return result

    def _delivery_summary_float(self, summary, key):
        fallback_keys = {
            "orders_completed": "agent_alive",
            "healthy_agents": "agent_health",
            "mean_goal_distance": "enemy_health",
        }
        if key in summary:
            return float(summary.get(key, 0.0))
        fallback_key = fallback_keys.get(key)
        if fallback_key is not None:
            return float(summary.get(fallback_key, 0.0))
        return 0.0

    def _delivery_diagnostic_values(self, summary):
        return [
            float(summary.get(key, 0.0))
            for key in UAV_DELIVERY_DIAGNOSTIC_KEYS
        ]

    def _build_default_log_row(
        self,
        event,
        episode,
        timestep,
        train_step,
        progress,
        episode_idx,
        episode_reward,
        episode_steps,
        epsilon,
        summary,
        safety_loss="",
        eval_step="",
    ):
        return [
            event,
            episode,
            timestep if timestep is not None else "",
            train_step if train_step is not None else "",
            progress,
            episode_idx,
            float(episode_reward),
            episode_steps,
            epsilon,
            float(summary.get("win_rate", summary.get("win_tag", 0.0)))
            if self.args.map in UAV_COLLISION_MAPS or self.args.map in SMAC_MAPS
            else "",
            float(summary.get("collision_count", 0.0))
            if self.args.map in UAV_COLLISION_MAPS
            else "",
            float(summary.get("obstacle_collision_count", 0.0))
            if self.args.map in UAV_COLLISION_MAPS
            else "",
            float(summary.get("agent_collision_count", 0.0))
            if self.args.map in UAV_COLLISION_MAPS
            else "",
            safety_loss,
            self._summary_float(summary, "agent_health") if event == "EVAL" else "",
            self._summary_float(summary, "enemy_health") if event == "EVAL" else "",
            self._summary_float(summary, "agent_alive") if event == "EVAL" else "",
            eval_step,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ]

    def _build_delivery_log_row(
        self,
        event,
        episode,
        timestep,
        train_step,
        progress,
        episode_idx,
        episode_reward,
        episode_steps,
        epsilon,
        summary,
        safety_loss="",
        eval_episode_steps="",
    ):
        return [
            event,
            episode,
            timestep if timestep is not None else "",
            train_step if train_step is not None else "",
            progress,
            episode_idx,
            float(episode_reward),
            episode_steps,
            epsilon,
            float(summary.get("win_rate", summary.get("win_tag", 0.0))),
            self._delivery_summary_float(summary, "orders_completed"),
            self._delivery_summary_float(summary, "total_orders"),
            self._delivery_summary_float(summary, "active_orders"),
            self._delivery_summary_float(summary, "available_orders"),
            self._delivery_summary_float(summary, "picked_orders"),
            self._delivery_summary_float(summary, "idle_agents"),
            self._delivery_summary_float(summary, "powered_agents"),
            self._delivery_summary_float(summary, "depleted_agents"),
            self._delivery_summary_float(summary, "mean_energy"),
            self._delivery_summary_float(summary, "charging_agents"),
            self._delivery_summary_float(summary, "mean_goal_distance"),
            self._delivery_summary_float(summary, "healthy_agents"),
            self._summary_float(summary, "collision_count"),
            self._summary_float(summary, "obstacle_collision_count"),
            self._summary_float(summary, "agent_collision_count"),
            safety_loss,
            eval_episode_steps,
            *self._delivery_diagnostic_values(summary),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ]

    def _build_log_row(
        self,
        event,
        episode,
        timestep,
        train_step,
        progress,
        episode_idx,
        episode_reward,
        episode_steps,
        epsilon,
        summary,
        safety_loss="",
        eval_step="",
    ):
        if self.args.map in UAV_DELIVERY_MAPS:
            return self._build_delivery_log_row(
                event,
                episode,
                timestep,
                train_step,
                progress,
                episode_idx,
                episode_reward,
                episode_steps,
                epsilon,
                summary,
                safety_loss=safety_loss,
                eval_episode_steps=eval_step,
            )
        return self._build_default_log_row(
            event,
            episode,
            timestep,
            train_step,
            progress,
            episode_idx,
            episode_reward,
            episode_steps,
            epsilon,
            summary,
            safety_loss=safety_loss,
            eval_step=eval_step,
        )

    def _log_eval_summary(self, summary, time_steps=None, train_steps=None):
        progress = ""
        if time_steps is not None and getattr(self.args, "n_steps", 0):
            progress = f"{100.0 * min(time_steps, self.args.n_steps) / self.args.n_steps:.2f}%"
        self.eval_count += 1
        self.last_eval_summary = dict(summary)
        self.last_eval_time_steps = time_steps
        self.last_eval_train_steps = train_steps
        self._append_log_row(
            self._build_log_row(
                "EVAL",
                self.eval_count,
                time_steps,
                train_steps,
                progress,
                "",
                float(summary.get("episode_reward", 0.0)),
                "",
                "",
                summary,
                eval_step=float(summary.get("step", 0.0)),
            )
        )

    def _init_agents(self): # for RGMComm
        agents = []
        for i in range(self.args.n_agents):
            agent = Agents(self.args, self.env, i)
            agents.append(agent)
        return agents

    def _save_policy_if_needed(self, time_steps, force=False):
        if not hasattr(self.agents, "policy"):
            return
        saver = None
        if hasattr(self.agents, "save_model"):
            saver = self.agents.save_model
        elif hasattr(self.agents.policy, "save_model"):
            saver = self.agents.policy.save_model
        if saver is None:
            return
        if time_steps is None:
            return

        save_cycle = getattr(self.args, "save_cycle", 0)
        current_bucket = (
            time_steps // save_cycle if save_cycle and time_steps > 0 else -1
        )
        should_save = force
        if not should_save:
            should_save = (
                save_cycle > 0
                and current_bucket > 0
                and current_bucket > self._last_saved_bucket
            )

        if not should_save:
            return

        saver(time_steps)
        self._save_cbf_flow_artifacts(train_step=int(time_steps), final=force)
        if current_bucket >= 0:
            self._last_saved_bucket = current_bucket

    def run(self, num):
        time_steps, train_steps, evaluate_steps = 0, 0, -1
        self._save_policy_if_needed(time_steps, force=True)
        try:
            with tqdm(
                total=self.args.n_steps,
                dynamic_ncols=True,
                ascii=True,
            ) as progress_bar:
                while time_steps < self.args.n_steps:
                    if time_steps // self.args.evaluate_cycle > evaluate_steps:
                        self._eval(num, time_steps=time_steps, train_steps=train_steps)
                        evaluate_steps += 1
                    episodes = []
                    pending_train_rows = []
                    if self.parallel_episode_collector is not None:
                        collected = self.parallel_episode_collector.collect_episodes(
                            self.args.n_episodes
                        )
                        rollout_epsilon = float(self.parallel_episode_collector.epsilon)
                        for episode_idx, result in enumerate(collected):
                            episode = result["episode"]
                            episode_reward = result["episode_reward"]
                            episode_summary = result["summary"]
                            steps = int(result["steps"])
                            episodes.append(episode)
                            time_steps += steps
                            progress_bar.update(
                                min(steps, max(self.args.n_steps - progress_bar.n, 0))
                            )
                            self.train_episode_count += 1
                            progress = (
                                f"{100.0 * min(time_steps, self.args.n_steps) / self.args.n_steps:.2f}%"
                                if getattr(self.args, "n_steps", 0)
                                else ""
                            )
                            pending_train_rows.append(
                                self._build_log_row(
                                    "TRAIN",
                                    self.train_episode_count,
                                    time_steps,
                                    train_steps,
                                    progress,
                                    episode_idx,
                                    float(episode_reward),
                                    steps,
                                    rollout_epsilon,
                                    episode_summary,
                                )
                            )
                            if time_steps >= self.args.n_steps:
                                break
                    else:
                        for episode_idx in range(self.args.n_episodes):
                            episode, episode_reward, episode_summary, steps, _ = self.rolloutWorker.generate_episode(  # type: ignore
                                episode_idx
                            )
                            episodes.append(episode)
                            time_steps += steps
                            progress_bar.update(
                                min(steps, max(self.args.n_steps - progress_bar.n, 0))
                            )
                            self.train_episode_count += 1
                            progress = (
                                f"{100.0 * min(time_steps, self.args.n_steps) / self.args.n_steps:.2f}%"
                                if getattr(self.args, "n_steps", 0)
                                else ""
                            )
                            pending_train_rows.append(
                                self._build_log_row(
                                    "TRAIN",
                                    self.train_episode_count,
                                    time_steps,
                                    train_steps,
                                    progress,
                                    episode_idx,
                                    float(episode_reward),
                                    steps,
                                    float(self.rolloutWorker.epsilon),
                                    episode_summary,
                                )
                            )
                            if time_steps >= self.args.n_steps:
                                break
                    episode_batch = episodes[0]
                    episodes.pop(0)
                    for episode in episodes:
                        for key in episode_batch.keys():
                            episode_batch[key] = np.concatenate(
                                (episode_batch[key], episode[key]), axis=0
                            )
                    self._drain_correction_records()
                    self._drain_energy_records()
                    self._drain_order_records()
                    phase = getattr(self.args, "training_phase", "low_cbf")
                    if (
                        self.args.alg.find("coma") > -1
                        or self.args.alg.find("central_v") > -1
                        or self.args.alg.find("reinforce") > -1
                        or self.args.alg.find("ippo") > -1
                        or self.args.alg.find("mappo") > -1
                        or self.args.alg.find("macpo") > -1
                    ):
                        if self.args.alg == "hmappo_cbf_flow" and phase in {"flow", "distill"}:
                            train_metrics = self._train_cbf_flow_phase()
                        else:
                            train_metrics = self.agents.train(
                                episode_batch, train_steps, self.rolloutWorker.epsilon
                            )
                            if self.args.alg == "hmappo_cbf_flow":
                                train_metrics.update(self._train_cbf_flow_phase())
                        safety_loss = (
                            train_metrics.get("safety_loss", "")
                            if isinstance(train_metrics, dict)
                            else ""
                        )
                        for row in pending_train_rows:
                            self._set_log_row_value(row, "safety_loss", safety_loss)
                            if isinstance(train_metrics, dict):
                                for metric_key, metric_value in train_metrics.items():
                                    self._set_log_row_value(
                                        row, metric_key, metric_value
                                    )
                            self._append_log_row(row)
                        train_steps += 1
                        self._save_policy_if_needed(time_steps)
                    else:
                        for row in pending_train_rows:
                            self._append_log_row(row)
                        self.buffer.store_episode(episode_batch)
                        for train_step in range(self.args.train_steps):
                            mini_batch = self.buffer.sample(
                                min(self.buffer.current_size, self.args.batch_size)
                            )
                            self.agents.train(mini_batch, train_steps)
                            train_steps += 1
                        self._save_policy_if_needed(time_steps)

            self._eval(num, time_steps=time_steps, train_steps=train_steps)
            self._save_policy_if_needed(time_steps, force=True)
            self._append_uav_delivery_experiment_result(time_steps, train_steps)
        finally:
            self.close()


    def _eval(self, num, time_steps=None, train_steps=None):
        if self.args.map == "Basic2P":
            summary, warning_signal = self.evaluate_maze()
            self.warning_signals.append(warning_signal)
            for key in summary.keys():
                self.maze_summary[key].append(summary[key])
            self.plt_maze()
            self._log_eval_summary(summary, time_steps=time_steps, train_steps=train_steps)
        elif self.args.map == "IoV":
            summary = self.evaluate_IoV()
            for key in summary.keys():
                self.IoV_summary[key].append(summary[key])
            self.plot_IoV()
            self._log_eval_summary(summary, time_steps=time_steps, train_steps=train_steps)
        elif self.args.map == "stego":
            summary = self.evaluate_stego()
            for metric in self.stego_summary.keys():
                self.stego_summary[metric].append(summary[metric])
                for agent in range(self.env.n_agents):
                        self.summary_agent[f'{agent}'][metric].append(
                            summary[f'{agent}'][metric]
                        )
            self.plot_stego()
            self._log_eval_summary(summary, time_steps=time_steps, train_steps=train_steps)
        else:  # smac
            win_rate, episode_reward, summary = self.evaluate()
            # print("win_rate is ", win_rate)
            self.win_rates.append(win_rate)
            self.episode_rewards.append(episode_reward)
            for key in summary.keys():
                self.smac_summary.setdefault(key, []).append(summary[key])
            if self.args.map not in UAV_DELIVERY_MAPS:
                self.plt(num)
                self.plt_smac(num, self.args)
            self._log_eval_summary(summary, time_steps=time_steps, train_steps=train_steps)

    def add_metrics(self):
        metrics = []
        if self.args.map == "Basic2P":
            metrics = self.maze_summary.keys()
        elif self.args.map == "IoV":
            metrics = self.IoV_summary.keys()
        elif self.args.map == "stego":
            metrics = self.stego_summary.keys()
        else:
            metrics = self.smac_summary.keys()
        for agent in range(self.env.n_agents):
            for metric in metrics:
                self.summary_agent[f'{agent}'][metric] = []

    def evaluate(self):
        # win_number = 0
        # episode_rewards = 0

        if self.args.map in UAV_DELIVERY_MAPS:
            summary = {
                "episode_reward": 0,
                "win_rate": 0,
                "collision_count": 0,
                "obstacle_collision_count": 0,
                "agent_collision_count": 0,
                "step": 0,
            }
            for key in UAV_DELIVERY_SUMMARY_KEYS:
                summary[key] = 0
            for key in UAV_DELIVERY_DIAGNOSTIC_KEYS:
                summary[key] = 0
        else:
            summary = {
                "episode_reward": 0,
                "win_rate": 0,
                "collision_count": 0,
                "obstacle_collision_count": 0,
                "agent_collision_count": 0,
                "step": 0,
                "agent_health": 0,
                "enemy_health": 0,
                "agent_alive": 0,
            }

        for epoch in range(self.args.evaluate_epoch):
            # _, episode_reward, win_tag, _ = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            # episode_rewards += episode_reward
            # if episode_summary['win_tag']:
            # win_number += 1
            _, episode_reward, episode_summary, _, _ = (
                self.rolloutWorker.generate_episode(epoch, evaluate=True)
            )

            summary["episode_reward"] += episode_reward
            summary["win_rate"] += episode_summary["win_tag"]
            summary["collision_count"] += episode_summary.get("collision_count", 0.0)
            summary["obstacle_collision_count"] += episode_summary.get(
                "obstacle_collision_count", 0.0
            )
            summary["agent_collision_count"] += episode_summary.get(
                "agent_collision_count", 0.0
            )
            summary["step"] += episode_summary["step"]
            if self.args.map in UAV_DELIVERY_MAPS:
                summary["orders_completed"] += episode_summary.get(
                    "orders_completed", episode_summary.get("agent_alive", 0.0)
                )
                summary["total_orders"] += episode_summary.get("total_orders", 0.0)
                summary["active_orders"] += episode_summary.get("active_orders", 0.0)
                summary["available_orders"] += episode_summary.get(
                    "available_orders", 0.0
                )
                summary["picked_orders"] += episode_summary.get("picked_orders", 0.0)
                summary["idle_agents"] += episode_summary.get("idle_agents", 0.0)
                summary["mean_goal_distance"] += episode_summary.get(
                    "mean_goal_distance", episode_summary.get("enemy_health", 0.0)
                )
                summary["healthy_agents"] += episode_summary.get(
                    "healthy_agents", episode_summary.get("agent_health", 0.0)
                )
                for key in UAV_DELIVERY_DIAGNOSTIC_KEYS:
                    summary[key] += episode_summary.get(key, 0.0)
            else:
                summary["agent_health"] += episode_summary["agent_health"]
                summary["enemy_health"] += episode_summary["enemy_health"]
                summary["agent_alive"] += episode_summary["agent_alive"]

        for key in summary.keys():
            summary[key] /= self.args.evaluate_epoch

        return summary["win_rate"], summary["episode_reward"], summary

        # return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch

    def evaluate_maze(self):
        episode = {
            "reward": 0.0,
            "trap": 0.0,
            "bonus": 0.0,
            "life": 0.0,
            "step": 0.0,
            "arrived": 0.0,
            "died": 0.0,
            "warning_signal": 0.0,
        }
        for i in range(self.env.n_agents):
            episode.update({f"agent{i}_trap": 0.0, f"agent{i}_life": 0.0})

        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, episode_summary, _, warning_signal = (
                self.rolloutWorker.generate_episode(epoch, evaluate=True)
            )
            episode["reward"] += episode_reward
            episode["trap"] += episode_summary["trap"]
            episode["bonus"] += episode_summary["bonus"]
            episode["life"] += episode_summary["life"]
            episode["step"] += episode_summary["step"]
            episode["arrived"] += episode_summary["arrived"]
            episode["died"] += episode_summary["died"]
            episode["warning_signal"] += sum(warning_signal)
            for i in range(self.env.n_agents):
                episode[f"agent{i}_trap"] += episode_summary[f"agent{i}_trap"]
                episode[f"agent{i}_life"] += episode_summary[f"agent{i}_life"]

        for key in episode.keys():
            episode[key] = float(episode[key]) / self.args.evaluate_epoch

        return episode, episode["warning_signal"]

    def evaluate_IoV(self):
        episode = {
            "reward": 0.0,
            "latency": 0.0,
            "protection_level": 0.0,
            "atk_succ_rate": 0.0,
            "malicious_msg_nums": 0.0,
        }
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, episode_summary, _, warning_signal = (
                self.rolloutWorker.generate_episode(epoch, evaluate=True)
            )
            episode["reward"] += episode_reward
            episode["latency"] += episode_summary["latency"]
            episode["protection_level"] += episode_summary["protection_level"]
            episode["atk_succ_rate"] += episode_summary["atk_succ_rate"]
            episode["malicious_msg_nums"] += episode_summary["malicious_msg_nums"]
        for key in episode.keys():
            episode[key] /= self.args.evaluate_epoch
        return episode

    def evaluate_stego(self):
        episode = {metric: 0.0 for metric in self.stego_summary.keys()}
        episode_agents = {}
        for agent in range(self.env.n_agents):
            episode_agents[f'{agent}'] = {}
            for metric in episode.keys():
                episode_agents[f'{agent}'][metric] = 0.0

        for epoch in range(self.args.evaluate_epoch):
            _, _, episode_summary, _, _ = self.rolloutWorker.generate_episode(
                epoch, evaluate=True
            )
            for key in episode.keys():
                episode[key] += episode_summary[key]
            for agent in range(self.env.n_agents):
                for metric in episode.keys():
                    episode_agents[f'{agent}'][metric] += episode_summary[f'{agent}'][metric]
        
        for key in episode.keys():
            episode[key] /= self.args.evaluate_epoch
            for agent in range(self.env.n_agents):
                for metric in episode.keys():
                    episode_agents[f'{agent}'][metric] /= self.args.evaluate_epoch
        episode.update(episode_agents)
        return episode

    def plt(self, num):
        plt.figure()
        plt.ylim([0, 105])
        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.win_rates)), self.win_rates)
        # plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel("win_rates")

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel("{} time steps".format(self.args.evaluate_cycle))
        plt.ylabel("episode_rewards")

        # plt.savefig(self.save_path + "/plt_{}.png".format(num), format="png")
        # np.save(self.save_path + "/win_rates_{}".format(num), self.win_rates)
        # np.save(
        #     self.save_path + "/episode_rewards_{}".format(num), self.episode_rewards
        # )
        plt.close()

    def plt_smac(self, num, args):
        plt.style.use("ggplot")
        fontsize = 12

        metrics = list(self.smac_summary.keys())
        n_metrics = len(metrics)
        n_cols = 3
        n_rows = int(np.ceil(n_metrics / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols)
        fig.suptitle(f"{self.args.alg.upper()}")
        axes = np.atleast_1d(axes).reshape(-1)

        for i, key in enumerate(metrics):
            axes[i].plot(self.smac_summary[key][1:])
            if i >= n_metrics - n_cols:
                axes[i].set_xlabel(
                    "{} time steps".format(self.args.evaluate_cycle), fontsize=fontsize
                )
            axes[i].set_title(key.title(), fontsize=fontsize)
            # np.save(self.save_path + "/{}_{}".format(key, num), self.smac_summary[key])

        for i in range(n_metrics, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        # plt.savefig(
        #     self.save_path + "/plt_plus_{}.png".format(num), format="png", dpi=300
        # )
        plt.close()

        alg_ = (
            self.args.alg
            if self.args.alg not in ["gmix_reshape_Comm", "gmix_Comm_reshape"]
            else "ours"
        )

        sio.savemat(
            self.save_path + f"{self.args.map}_{alg_}_{self.args.now}.mat",
            self.smac_summary,
        )

    def plt_maze(
        self,
    ):
        alg = self.args.alg
        sio.savemat(
            self.save_path + f"maze_{alg}_{self.args.now}.mat", self.maze_summary
        )
        self.agents.policy.save_model(self.args.n_steps // self.args.evaluate_cycle)
        # plt.style.use('bmh')
        # # 定义要删除的键列表
        # keys_to_remove = []
        # for key in range(self.env.n_agents):
        #     keys_to_remove.append(f'agent{key}_trap')
        #     keys_to_remove.append(f'agent{key}_life')

        # agent_single_perf = {key: value for key, value in self.maze_summary.items() if key in keys_to_remove}

        # # # analysis of every agent
        # # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # # fig.suptitle(f'{self.args.alg.upper()}')
        # # axes = axes.reshape(-1)
        # # for i, key in enumerate(['trap', 'life']):
        # #     for agent in range(self.env.n_agents):
        # #         axes[i].plot(agent_single_perf[f'agent{agent}_{key}'])
        # #     axes[i].set_title(key.title(), fontsize=12)
        # # plt.savefig(self.save_path + '/sole_{}.png'.format(num), format='png', dpi=300)
        # # plt.close()

        # fontsize = 12
        # subfig_num = len(self.maze_summary.keys()) - 4
        # row, col = 0, 0
        # for row in [2, 3, 5, 1]:
        #     if subfig_num % row == 0:
        #         col = subfig_num // row
        #         break
        # fig, axes = plt.subplots(row, col)
        # fig.suptitle(f'{self.args.alg.upper()}')
        # axes = axes.reshape(-1)

        # for i, key in enumerate(self.maze_summary.keys() - agent_single_perf.keys()):
        #     axes[i].plot(self.maze_summary[key])
        #     if i == 4:
        #         axes[i].set_xlabel('{} time steps'.format(self.args.evaluate_cycle), fontsize=fontsize)
        #     axes[i].set_title(key.title(), fontsize=fontsize)
        #     np.save(self.save_path + '/{}_{}'.format(key, num), self.maze_summary[key])

        # plt.savefig(self.save_path + '/plt_{}.png'.format(num), format='png', dpi=300)
        # plt.close()

    def plot_IoV(self):
        fig, axes = plt.subplots(2, 3)
        fig.suptitle(f"{self.args.alg.upper()}")
        axes = axes.reshape(-1)
        titles = [
            "Reward",
            "Latency (ms)",
            "Protection level",
            "Attack success rate (%)",
            "Malicious messages",
        ]

        for i, (key, title) in enumerate(
            zip(
                self.IoV_summary.keys(),
                titles,
            )
        ):
            axes[i].plot(self.IoV_summary[key][1:])
            if i == 4:
                axes[i].set_xlabel("{} time steps".format(self.args.evaluate_cycle))
            axes[i].set_title(title)

        sio.savemat(
            self.save_path + f"IoV_{self.args.alg}_{self.args.now}.mat", self.IoV_summary
        )
        plt.savefig(self.save_path + f"/IoV_{self.args.alg}.png", format="png", dpi=500)
        plt.close()

    def plot_stego(self):
        sio.savemat(
            self.save_path + f"stego_{self.args.alg}_{self.args.now}.mat",
            self.stego_summary,
        )
        for agent in range(self.env.n_agents):
            sio.savemat(
                self.save_path + f"stego_agent{agent}_{self.args.alg}_{self.args.now}.mat",
                self.summary_agent[f'{agent}'],
            )

    def runRGM(self):
        returns = []
        sample_dic = {}
        self.args.n_players = self.args.n_agents
        for i in range(self.args.n_agents):
            sample_dic[i] = []
        
        # 根据环境类型初始化结果字典
        result = {}
        result_episode = {}
        if self.args.map == "Basic2P":
            result = {
                'reward': [],
                'arrived': [],
                'died': [],
                'bonus': [],
                'trap': [],
                'life': [],
                'step': [],
                'warning_signal': []
            }
            result_episode = {
                'reward': [],
                'arrived': 0,
                'died': 0,
                'bonus': 0,
                'trap': 0,
                'life': 0,
                'step': 0,
                'warning_signal': 0
            }
        else:  # SMAC
            result = {
                'reward': [],
                'win_rate': []
            }
            result_episode = {
                'reward': [],
                'episode_won': False  # 记录本 episode 是否获胜
            }
        done = False
        last_start = 0
        episode_count = 0  # 记录 episode 数量
        win_history = []  # 记录最近 N 个 episode 的胜负历史
        episode_count = 0  # 记录 episode 数量
        win_history = []  # 记录最近 N 个 episode 的胜负历史
        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            if time_step % self.args.max_episode_len == 0 or done:
                # 先记录上一个 episode 的结果（如果不是第一次）
                if time_step > 0:
                    if self.args.map == "Basic2P":
                        result['reward'].append(sum(result_episode['reward']))
                        result['arrived'].append(result_episode['arrived'])
                        result['died'].append(result_episode['died'])
                        result['bonus'].append(result_episode['bonus'])
                        result['trap'].append(result_episode['trap'])
                        result['life'].append(result_episode['life'])
                        result['step'].append(result_episode['step'])
                        result['warning_signal'].append(result_episode['warning_signal'])
                    else:  # SMAC
                        result['reward'].append(sum(result_episode['reward']))
                        # 记录本 episode 是否获胜（0 或 1）
                        episode_won = 1 if result_episode['episode_won'] else 0
                        win_history.append(episode_won)
                        episode_count += 1
                        
                        # 计算滑动窗口平均 win rate（最近 100 个 episode）
                        window_size = min(100, episode_count)
                        recent_wins = win_history[-window_size:]
                        avg_win_rate = sum(recent_wins) / len(recent_wins)
                        result['win_rate'].append(avg_win_rate)
                
                # 重置环境和 episode 记录
                self._reset_rgm_env_for_episode(evaluate=False)
                s = self.env.get_obs()
                # 更新结果，最终记录的是一个 episode 的
                if self.args.map == "Basic2P":
                    result_episode = {
                        'reward': [],
                        'arrived': 0,
                        'died': 0,
                        'bonus': 0,
                        'trap': 0,
                        'life': 0,
                        'step': 0,
                        'warning_signal': 0
                    }
                else:  # SMAC
                    result_episode = {
                        'reward': [],
                        'episode_won': False
                    }
                last_start = time_step
            u = []
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    # import pdb; pdb.set_trace()
                    action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                    u.append(action)
                    actions.append(action)
            for i in range(self.args.n_agents, self.args.n_players):
                actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])

            # action_index = []
            # for agent_id, action in enumerate(actions):
            #     available_actions = self.env.get_avail_agent_actions(agent_id)
            #     available_ids = []
            #     for i, a_action in enumerate(available_actions):
            #         if a_action == 1:
            #             available_ids.append(i)
            #     max_val_action = np.max(action[available_ids])
            #     action_i = action.tolist().index(max_val_action)
            #     action_index.append(action_i)

            action_index = []
            for agent_id, action in enumerate(actions):
                # 1. 获取当前智能体的可用动作
                available_actions = self.env.get_avail_agent_actions(agent_id)
                available_ids = np.where(np.array(available_actions) == 1)[0]  # 可用动作的索引
                
                # 2. 检查是否有可用动作
                if len(available_ids) == 0:
                    action_index.append(8)  # 默认动作（如无操作）
                    continue
                
                # 3. 在可用动作中选择概率最大的动作
                available_action_probs = action[available_ids]  # 仅保留可用动作的概率
                max_prob = np.max(available_action_probs)      # 最大概率值
                # 可能有多个动作具有相同最大概率，随机选一个
                candidate_actions = available_ids[available_action_probs == max_prob]
                chosen_action = np.random.choice(candidate_actions)  # 随机选一个
                action_index.append(chosen_action)

            # import pdb; pdb.set_trace()
            # s_next, r, done, info = self.env.step(actions)
            r, done, info = self.env.step(action_index)

            # 针对环境修正 reward 标准化
            if self.args.map == "Basic2P":
                r *= 0.005

            result_episode['reward'].append(r)
            if self.args.map == "Basic2P":
                result_episode['arrived'] = info.get('arrived', 0)  # 最终状态，用 = 覆盖
                result_episode['died'] = info.get('died', 0)  # 最终状态，用 = 覆盖
                result_episode['bonus'] += sum(info.get('bonus', [0]))  # 累计值，用 +=
                result_episode['trap'] = sum(info.get('traps', [0]))  # 每步覆盖为当前值（最后一步是总数）
                result_episode['life'] = sum(info.get('life', [0]))  # 最终状态，用 = 覆盖（记录当前生命值）
                result_episode['step'] += 1  # 增加 step 计数
                result_episode['warning_signal'] += np.sum(info.get('warning_signal', 0))  # 累计值，用 +=
            else:  # SMAC
                # 只要本 episode 中任意一步 battle_won 为 True，就记为获胜
                if info.get('battle_won', False):
                    result_episode['episode_won'] = True
            
            rewards = [r for _ in range(self.args.n_agents)]
            s_next = []
            if hasattr(self.env, "get_obs"):
                s_next = self.env.get_obs()
            else:
                for agent_id in range(self.args.n_agents):
                    s_next.append(self.env.get_obs_agent(agent_id))
            # import pdb; pdb.set_trace()
            self.buffer.store_episode(s[:self.args.n_agents], action_index[:self.args.n_agents], rewards[:self.args.n_agents], s_next[:self.args.n_agents])
            s = s_next
            if self.buffer.current_size >= self.args.batch_size:
                transitions = self.buffer.sample(self.args.batch_size)
                for agent_idx, agent in enumerate(self.agents):
                    other_agents = self.agents[:agent_idx] + self.agents[agent_idx+1:]
                    agent.learn(transitions, other_agents)

            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                returns.append(self.evaluateRGM())

            if time_step >= self.args.sample_start and time_step % self.args.sample_rate == 0:
                transitions = self.buffer.sample(self.args.batch_size)
                for agent in self.agents:
                    sample_dic[agent.agent_id].extend(agent.learnit(transitions))
            self.noise = max(0.05, self.noise - 0.0000005)
            self.epsilon = max(0.05, self.noise - 0.0000005)
        
        # 训练结束后保存结果
        sio.savemat(
            self.save_path + f"{self.args.map}_rgmcomm_{self.args.now}.mat",
            result,
        )
        self._append_uav_delivery_rgm_experiment_result(
            result,
            result_episode,
            int(getattr(self.args, "time_steps", 0)),
        )
        
        # 保存评估曲线
        if len(returns) > 0:
            plt.figure()
            plt.plot(range(len(returns)), returns)
            plt.xlabel('Evaluation episodes')
            plt.ylabel('Average returns')
            plt.savefig(self.save_path + '/plt.png', format='png', dpi=300)
            plt.close()
        # np.save(self.save_path + '/returns.pkl', returns)
        # for agent_id in sample_dic.keys():
        #     pd.DataFrame(sample_dic[agent_id]).to_csv(self.args.save_dir + '/' + self.args.scenario_name + '/QTableStage1/tag_6_{}_{}_{}.csv'.format(self.args.sample_start, self.args.time_steps, agent_id))

    def evaluateRGM(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            seed = derive_episode_seed(
                self.args,
                evaluate=True,
                episode_index=episode,
            )
            with temporary_seed(seed, include_torch=True):
                rewards = self._evaluate_rgm_episode(episode)
            returns.append(rewards)
            print("Returns is", rewards)
        return sum(returns) / max(1, self.args.evaluate_episodes)

    def _evaluate_rgm_episode(self, episode):
        self._reset_rgm_env_for_episode(evaluate=True, episode_index=episode)
        s = self.env.get_obs()
        rewards = 0
        for time_step in range(self.args.evaluate_episode_len):
            if hasattr(self.env, "render"):
                try:
                    self.env.render()
                except NotImplementedError:
                    pass
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    action = agent.select_action(s[agent_id], 0, 0)
                    actions.append(action)

            action_index = []
            for agent_id, action in enumerate(actions):
                available_actions = self.env.get_avail_agent_actions(agent_id)
                available_ids = np.where(np.array(available_actions) == 1)[0]

                if len(available_ids) == 0:
                    action_index.append(8)
                    continue

                available_action_probs = action[available_ids]
                max_prob = np.max(available_action_probs)
                candidate_actions = available_ids[available_action_probs == max_prob]
                chosen_action = np.random.choice(candidate_actions)
                action_index.append(chosen_action)

            for i in range(self.args.n_agents, self.args.n_players):
                actions.append(
                    [0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0]
                )
            r, done, info = self.env.step(action_index)

            if self.args.map == "Basic2P":
                r *= 0.005

            if hasattr(self.env, "get_obs"):
                s_next = self.env.get_obs()
            else:
                s_next = []
                for agent_id in range(self.args.n_agents):
                    s_next.append(self.env.get_obs_agent(agent_id))
            s = s_next
            rewards += r
            s = s_next
        return rewards


if __name__ == "__main__":
    pass
