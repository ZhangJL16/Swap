import os
from datetime import datetime
from types import MethodType

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from common.arguments import (
    get_centralv_args,
    get_coma_args,
    get_commnet_args,
    get_g2anet_args,
    get_macpo_args,
    get_mappo_args,
    get_mixer_args,
    get_reinforce_args,
    get_RGM_args,
    get_common_args,
)
from common.auction import auction_assign_min_cost
from common.seeding import seed_everything
from main import build_env
from runner import Runner


def _base_energy_delivery_env(env):
    return getattr(env, "env", env)


def install_auction_assignment(env):
    """Replace UAVEnergyDelivery's nearest-order assignment with auction assignment."""
    base_env = _base_energy_delivery_env(env)
    required = (
        "_activate_orders",
        "_available_orders",
        "_assign_order_to_agent",
        "_set_agent_idle",
        "_sync_goals_from_orders",
    )
    missing = [name for name in required if not hasattr(base_env, name)]
    if missing:
        raise AttributeError(
            "Auction assignment requires UAVEnergyDelivery internals; "
            f"missing: {', '.join(missing)}"
        )

    def _auction_assign_orders(self):
        self._activate_orders()
        candidates = [
            agent
            for agent in self.agents
            if agent.assigned_order_id is None
            and (not hasattr(agent, "has_energy") or agent.has_energy())
        ]
        available_orders = self._available_orders()
        if not candidates:
            self._sync_goals_from_orders()
            return
        if not available_orders:
            for agent in candidates:
                self._set_agent_idle(agent)
            self._sync_goals_from_orders()
            return

        cost_matrix = np.zeros((len(candidates), len(available_orders)), dtype=np.float32)
        for agent_idx, agent in enumerate(candidates):
            for order_idx, order in enumerate(available_orders):
                pickup_dist = float(np.linalg.norm(order.pickup_pos - agent.pos))
                delivery_dist = float(np.linalg.norm(order.dropoff_pos - order.pickup_pos))
                cost_matrix[agent_idx, order_idx] = pickup_dist + delivery_dist

        assignments = auction_assign_min_cost(cost_matrix)
        assigned_agents = set()
        for agent_idx, order_idx in assignments.items():
            if order_idx >= len(available_orders):
                continue
            agent = candidates[agent_idx]
            order = available_orders[order_idx]
            self._assign_order_to_agent(agent, order)
            assigned_agents.add(agent_idx)

        for agent_idx, agent in enumerate(candidates):
            if agent_idx not in assigned_agents:
                self._set_agent_idle(agent)
        self._sync_goals_from_orders()

    base_env._assign_orders = MethodType(_auction_assign_orders, base_env)
    return env


def configure_algorithm_args(args):
    if args.alg.find("coma") > -1:
        return get_coma_args(args)
    if args.alg.find("central_v") > -1:
        return get_centralv_args(args)
    if args.alg.find("reinforce") > -1:
        return get_reinforce_args(args)
    if args.alg.find("mappo") > -1:
        return get_mappo_args(args)
    if args.alg.find("macpo") > -1:
        return get_macpo_args(args)
    if args.alg.lower().find("rgmcomm") > -1:
        return get_RGM_args(args)
    return get_mixer_args(args)


def apply_env_info(args, env):
    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.raw_obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]
    args.msg_shape = env_info.get("msg_shape", 0)
    return args


def main():
    args = get_common_args()
    args.now = datetime.now().strftime("%m%d_%H%M%S")
    if args.map != "UAVEnergyDelivery":
        raise ValueError(
            "main_energy_delivery_auction.py is scoped to --map UAVEnergyDelivery"
        )
    if args.alg.lower() != "mappo":
        raise ValueError("Auction assignment training script currently supports --alg mappo")

    seed_everything(
        args.seed,
        deterministic_torch=bool(getattr(args, "deterministic_torch", True)),
    )
    if args.cuda and torch.cuda.is_available():
        torch.cuda.set_device(int(getattr(args, "gpu_id", 0)))

    env = build_env(args, [args.alg])
    install_auction_assignment(env)
    args = apply_env_info(args, env)
    args = configure_algorithm_args(args)
    args.run_script = os.path.basename(__file__)
    runner = Runner(env, args)
    runner.run(0)
    env.close()


if __name__ == "__main__":
    main()
