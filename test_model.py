import argparse
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from agent.agent import Agents
from common.arguments import (
    get_centralv_args,
    get_coma_args,
    get_commnet_args,
    get_g2anet_args,
    get_macpo_args,
    get_mappo_args,
    get_mixer_args,
    get_reinforce_args,
)
from common.rollout import _build_env_summary, _get_active_agent_mask, _get_env_msg
from main_level import build_env
from safety.distilled_corrector import DistilledCorrector


def positive_int(value):
    int_value = int(value)
    if int_value <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return int_value


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a trained model and run evaluation episodes."
    )
    parser.add_argument("--alg", default="mappo", help="Algorithm name")
    parser.add_argument("--map", default="UAV3D", help="Environment map name")
    parser.add_argument(
        "--episodes", type=positive_int, default=1, help="Number of test episodes"
    )
    parser.add_argument(
        "--max-steps",
        type=positive_int,
        default=None,
        help="Optional per-episode step cap. Defaults to env episode_limit.",
    )
    parser.add_argument(
        "--model-dir", default="./model", help="Root directory of saved checkpoints"
    )
    parser.add_argument(
        "--result-root",
        default="./result/test",
        help="Directory for saved evaluation GIFs",
    )
    parser.add_argument(
        "--frame-root",
        default="./test_result",
        help="Directory for saved per-frame 3D and XY-view images",
    )
    parser.add_argument(
        "--render-mode",
        default="rgb_array",
        choices=["none", "rgb_array", "human"],
        help="Visualization mode",
    )
    parser.add_argument(
        "--render",
        dest="render",
        action="store_true",
        help="Enable visualization",
    )
    parser.add_argument(
        "--no-render",
        dest="render",
        action="store_false",
        help="Disable visualization",
    )
    parser.add_argument(
        "--gif-fps",
        type=positive_int,
        default=8,
        help="Playback FPS for saved GIFs",
    )
    parser.add_argument(
        "--fps",
        type=positive_int,
        default=20,
        help="Refresh rate for human rendering",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed. Only used if the environment reset supports it.",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA if the loaded policy supports it",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU id to use when CUDA is enabled",
    )
    parser.add_argument(
        "--uav-n-agents",
        "--uav_n_agents",
        dest="uav_n_agents",
        type=positive_int,
        default=4,
        help="Number of UAV agents for UAV2D/UAV3D environments",
    )
    parser.add_argument(
        "--episode-limit",
        "--episode_limit",
        dest="episode_limit",
        type=positive_int,
        default=400,
        help="Episode length for local UAV environments",
    )
    parser.add_argument(
        "--uav-total-orders",
        "--uav_total_orders",
        dest="uav_total_orders",
        type=positive_int,
        default=16,
        help="Total delivery orders per UAVDelivery/UAVEnergyDelivery episode",
    )
    parser.add_argument(
        "--uav-max-active-orders",
        "--uav_max_active_orders",
        dest="uav_max_active_orders",
        type=positive_int,
        default=8,
        help="Maximum simultaneously active delivery orders",
    )
    parser.add_argument(
        "--uav-initial-energy",
        "--uav_initial_energy",
        dest="uav_initial_energy",
        type=float,
        default=100.0,
        help="Initial energy for UAVEnergyDelivery agents",
    )
    parser.add_argument(
        "--uav-energy-decay",
        "--uav_energy_decay",
        dest="uav_energy_decay",
        type=float,
        default=None,
        help="Energy consumed per UAVEnergyDelivery step",
    )
    parser.add_argument(
        "--uav-energy-depletion-fraction",
        "--uav_energy_depletion_fraction",
        dest="uav_energy_depletion_fraction",
        type=float,
        default=0.5,
        help="Fraction of episode length when full energy should be depleted",
    )
    parser.add_argument(
        "--uav-charging-capacity",
        "--uav_charging_capacity",
        dest="uav_charging_capacity",
        type=positive_int,
        default=2,
        help="Maximum UAVs charging at the station in one step",
    )
    parser.add_argument(
        "--uav-charging-radius",
        "--uav_charging_radius",
        dest="uav_charging_radius",
        type=float,
        default=0.18,
        help="Distance threshold for UAVEnergyDelivery charging station",
    )
    parser.add_argument(
        "--uav-charging-rate",
        "--uav_charging_rate",
        dest="uav_charging_rate",
        type=float,
        default=None,
        help="Energy restored per charging step; defaults to 4x energy decay per step",
    )
    parser.add_argument(
        "--hmappo-meta-period",
        "--hmappo_meta_period",
        dest="hmappo_meta_period",
        type=positive_int,
        default=5,
        help="Low-level steps between high-level HMAPPO decisions",
    )
    parser.add_argument(
        "--hrl-reachable-subgoal-scale",
        "--hrl_reachable_subgoal_scale",
        dest="hrl_reachable_subgoal_scale",
        type=float,
        default=1.0,
        help="Scale applied to the reachable local subgoal radius",
    )
    parser.add_argument(
        "--hrl-intrinsic-reward-scale",
        "--hrl_intrinsic_reward_scale",
        dest="hrl_intrinsic_reward_scale",
        type=float,
        default=1.0,
        help="Scale for low-level HRL intrinsic rewards",
    )
    parser.add_argument(
        "--hrl-intrinsic-success-bonus",
        "--hrl_intrinsic_success_bonus",
        dest="hrl_intrinsic_success_bonus",
        type=float,
        default=1.0,
        help="Success bonus in the low-level intrinsic reward",
    )
    parser.add_argument(
        "--hrl-intrinsic-collision-penalty",
        "--hrl_intrinsic_collision_penalty",
        dest="hrl_intrinsic_collision_penalty",
        type=float,
        default=0.0,
        help="Low-level intrinsic penalty for a collision on the current step",
    )
    parser.add_argument(
        "--hrl-order-progress-override",
        "--hrl_order_progress_override",
        dest="hrl_order_progress_override",
        type=float,
        default=None,
        help="Override high-level progress scalar for order-mode subgoals",
    )
    parser.add_argument(
        "--hrl-energy-margin-reserve-ratio",
        "--hrl_energy_margin_reserve_ratio",
        dest="hrl_energy_margin_reserve_ratio",
        type=float,
        default=0.05,
        help="Normalized energy reserve after completing an order and returning",
    )
    parser.add_argument(
        "--hrl-charge-queue-enabled",
        "--hrl_charge_queue_enabled",
        dest="hrl_charge_queue_enabled",
        type=str2bool,
        default=False,
        help="Spread charging subgoals across dock and waiting slots",
    )
    parser.add_argument(
        "--hrl-charge-queue-radius",
        "--hrl_charge_queue_radius",
        dest="hrl_charge_queue_radius",
        type=float,
        default=0.24,
        help="Radius of the charging waiting ring",
    )
    parser.add_argument(
        "--hrl-charge-mode-fraction",
        "--hrl_charge_mode_fraction",
        dest="hrl_charge_mode_fraction",
        type=float,
        default=0.5,
        help="Fraction of high-level mode interval reserved for charging",
    )
    parser.add_argument(
        "--hrl-charge-dense-reward-scale",
        "--hrl_charge_dense_reward_scale",
        dest="hrl_charge_dense_reward_scale",
        type=float,
        default=0.0,
        help="Dense goal-shaping scale when current high-level target is charging",
    )
    parser.add_argument(
        "--hrl-safe-action-guard-enabled",
        "--hrl_safe_action_guard_enabled",
        dest="hrl_safe_action_guard_enabled",
        type=str2bool,
        default=False,
        help="Enable SCOPE-style deterministic low-level safety action replacement",
    )
    parser.add_argument(
        "--hrl-safe-action-guard-margin",
        "--hrl_safe_action_guard_margin",
        dest="hrl_safe_action_guard_margin",
        type=float,
        default=0.04,
        help="Extra collision prediction margin used by the low-level safety action replacement",
    )
    parser.add_argument(
        "--agent-entry-interval",
        "--agent_entry_interval",
        dest="agent_entry_interval",
        type=positive_int,
        default=1,
        help="hmappo_cbf_flow UAV entry interval",
    )
    parser.add_argument(
        "--order-max-duration",
        "--order_max_duration",
        dest="order_max_duration",
        type=positive_int,
        default=120,
        help="hmappo_cbf_flow maximum option duration",
    )
    parser.add_argument(
        "--energy-reserve",
        "--energy_reserve",
        dest="energy_reserve",
        type=float,
        default=0.0,
        help="hmappo_cbf_flow energy reserve",
    )
    parser.add_argument(
        "--cbf-flow-artifact-dir",
        "--cbf_flow_artifact_dir",
        dest="cbf_flow_artifact_dir",
        type=str,
        default="",
        help="hmappo_cbf_flow artifact directory",
    )
    parser.add_argument(
        "--cbf-flow-use-student",
        "--cbf_flow_use_student",
        dest="cbf_flow_use_student",
        action="store_true",
        help="Use distilled student corrector during hmappo_cbf_flow evaluation",
    )
    parser.add_argument(
        "--save-xy",
        dest="save_xy",
        action="store_true",
        help="For UAV3D, save an XY-view GIF when using rgb_array render",
    )
    parser.add_argument(
        "--no-save-xy",
        dest="save_xy",
        action="store_false",
        help="Disable XY-view GIF output",
    )
    parser.add_argument(
        "--save-frames",
        dest="save_frames",
        action="store_true",
        help="Save each rendered 3D frame and top-view frame as PNG files",
    )
    parser.add_argument(
        "--no-save-frames",
        dest="save_frames",
        action="store_false",
        help="Disable per-frame PNG output",
    )
    parser.set_defaults(save_frames=True)
    parser.set_defaults(save_xy=True)
    parser.set_defaults(render=True)
    return parser.parse_args()


def make_base_args(cli_args):
    return SimpleNamespace(
        difficulty="7",
        game_version="latest",
        map=cli_args.map,
        sc2_path='D:/Program Files (x86)/StarCraft II',
        seed=123 if cli_args.seed is None else cli_args.seed,
        step_mul=8,
        replay_dir="",
        debug=False,
        alg=cli_args.alg,
        alg_list="",
        n_steps=1_000_000,
        n_episodes=1,
        evaluate_cycle=20,
        evaluate_epoch=1,
        last_action=False,
        reuse_network=True,
        gamma=0.99,
        optimizer="RMS",
        model_dir=cli_args.model_dir,
        result_dir="./result",
        load_model=True,
        evaluate=True,
        cuda=cli_args.cuda,
        gpu_id=cli_args.gpu_id,
        use_level_policy=(
            cli_args.map.startswith("UAVEnergyDeliveryLevel")
            or cli_args.alg.lower() in {"hmappo", "hmappo_cbf_flow"}
        ),
        training_phase="distill",
        agent_entry_interval=cli_args.agent_entry_interval,
        order_max_duration=cli_args.order_max_duration,
        energy_reserve=cli_args.energy_reserve,
        cbf_flow_artifact_dir=cli_args.cbf_flow_artifact_dir,
        cbf_flow_load_artifact_dir=cli_args.cbf_flow_artifact_dir,
        cbf_flow_export_student=False,
        cbf_flow_student_mse_threshold=0.02,
        flow_hidden_dim=256,
        flow_eval_steps=8,
        student_margin=0.0,
        is_level_training=(
            cli_args.map.startswith("UAVEnergyDeliveryLevel")
            or cli_args.alg.lower() == "hmappo_cbf_flow"
        ),
        hmappo_meta_period=cli_args.hmappo_meta_period,
        hrl_reachable_subgoal_scale=cli_args.hrl_reachable_subgoal_scale,
        hrl_intrinsic_reward_scale=cli_args.hrl_intrinsic_reward_scale,
        hrl_intrinsic_success_bonus=cli_args.hrl_intrinsic_success_bonus,
        hrl_intrinsic_collision_penalty=cli_args.hrl_intrinsic_collision_penalty,
        hrl_order_progress_override=cli_args.hrl_order_progress_override,
        hrl_energy_margin_reserve_ratio=cli_args.hrl_energy_margin_reserve_ratio,
        hrl_charge_queue_enabled=cli_args.hrl_charge_queue_enabled,
        hrl_charge_queue_radius=cli_args.hrl_charge_queue_radius,
        hrl_charge_mode_fraction=cli_args.hrl_charge_mode_fraction,
        hrl_charge_dense_reward_scale=cli_args.hrl_charge_dense_reward_scale,
        hrl_meta_update_on_subgoal_done=False,
        hrl_safe_action_guard_enabled=cli_args.hrl_safe_action_guard_enabled,
        hrl_safe_action_guard_margin=cli_args.hrl_safe_action_guard_margin,
        uav_n_agents=cli_args.uav_n_agents,
        episode_limit=cli_args.episode_limit,
        uav_total_orders=cli_args.uav_total_orders,
        uav_max_active_orders=cli_args.uav_max_active_orders,
        uav_initial_energy=cli_args.uav_initial_energy,
        uav_energy_decay=cli_args.uav_energy_decay,
        uav_energy_depletion_fraction=cli_args.uav_energy_depletion_fraction,
        uav_charging_capacity=cli_args.uav_charging_capacity,
        uav_charging_radius=cli_args.uav_charging_radius,
        uav_charging_rate=cli_args.uav_charging_rate,
        last_reward=False,
        distributed=True,
        guide_mix_network_type="vdn",
        comm=False,
        msg_size=3,
        aoi_threshold=0.25,
        comm_cost_penalty=0.1,
        comm_effect_bonus=0.1,
        comm_warning_threshold=0.1,
        comm_max_keep_dim=8,
        comm_lr=None,
        safety_lr=None,
        safety_beta=None,
        warning_penalty_weight=None,
        aoi_min_weight=0.2,
        aoi_stale_decay=0.5,
        comm_aoi_penalty=0.1,
        comm_warning_penalty=0.2,
        comm_fresh_bonus=0.05,
        now="test",
    )


def configure_algorithm_args(args):
    if args.alg.find("coma") > -1:
        args = get_coma_args(args)
    elif args.alg.find("central_v") > -1:
        args = get_centralv_args(args)
    elif args.alg.find("reinforce") > -1:
        args = get_reinforce_args(args)
    elif args.alg == "hmappo_cbf_flow":
        args.use_level_policy = True
        args.is_level_training = True
        args = get_mappo_args(args)
    elif args.alg.find("mappo") > -1:
        args = get_mappo_args(args)
    elif args.alg.find("macpo") > -1:
        args = get_macpo_args(args)
    else:
        args = get_mixer_args(args)

    if args.alg.find("commnet") > -1:
        args = get_commnet_args(args)
    if args.alg.find("g2anet") > -1:
        args = get_g2anet_args(args)

    if args.alg.lower().find("comm") > -1 and args.alg.lower().find("rgmcomm") < 0:
        args.msg_shape = min(
            int(getattr(args, "comm_max_keep_dim", args.raw_obs_shape)),
            int(args.raw_obs_shape),
        )
        args.obs_shape += args.msg_shape * (args.n_agents - 1)
        args.state_shape = args.obs_shape * args.n_agents

    return args


def reset_env(env, seed=None):
    if seed is None:
        return env.reset()

    try:
        return env.reset(seed=seed)
    except TypeError:
        print(
            "Warning: environment reset does not accept seed; "
            "initialization may not be reproducible."
        )
        return env.reset()


def set_episode_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def frame_dir(frame_root, alg, map_name, episode_idx, view):
    output_dir = os.path.join(
        frame_root,
        alg,
        map_name,
        f"episode_{episode_idx}",
        view,
    )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_frame_image(frame, frame_root, alg, map_name, episode_idx, view, frame_idx):
    output_dir = frame_dir(frame_root, alg, map_name, episode_idx, view)
    output_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
    Image.fromarray(frame).save(output_path)
    return output_path


def render_frame(
    env,
    render_mode,
    frames,
    fps,
    xy_frames=None,
    save_xy=False,
    save_frames=False,
    frame_root=None,
    alg=None,
    map_name=None,
    episode_idx=0,
    frame_idx=0,
):
    if render_mode == "none":
        return

    frame = None
    xy_frame = None
    if hasattr(env, "env") and hasattr(env.env, "render"):
        frame = env.env.render(show=(render_mode == "human"))
        if save_xy or save_frames:
            xy_frame = env.env.render(show=False, view="xy")
    elif hasattr(env, "render"):
        frame = env.render()

    if render_mode == "human":
        if fps > 0:
            time.sleep(1.0 / fps)
        return

    if frame is not None:
        frames.append(Image.fromarray(frame))
        if save_frames:
            save_frame_image(
                frame,
                frame_root,
                alg,
                map_name,
                episode_idx,
                "3d",
                frame_idx,
            )
    if save_xy and xy_frame is not None and xy_frames is not None:
        xy_frames.append(Image.fromarray(xy_frame))
    if save_frames and xy_frame is not None:
        save_frame_image(
            xy_frame,
            frame_root,
            alg,
            map_name,
            episode_idx,
            "xy",
            frame_idx,
        )


def gif_path(result_root, alg, map_name, episode_idx, suffix=""):
    result_dir = os.path.join(result_root, alg, map_name)
    os.makedirs(result_dir, exist_ok=True)
    suffix_part = f"_{suffix}" if suffix else ""
    return os.path.join(result_dir, f"{map_name}_episode_{episode_idx}{suffix_part}.gif")


def print_model_info(args):
    model_path = os.path.join(args.model_dir, args.alg, args.map)
    print(f"Loaded model: {args.alg}/{args.map}")
    print(f"Model directory: {os.path.abspath(model_path)}")


def choose_actions(env, agents, args, last_action, step):
    raw_obs = np.asarray(env.get_obs(), dtype=np.float32)
    obs = raw_obs
    if getattr(agents, "use_comm_plugin", False) and args.alg.lower().find("rgmcomm") < 0:
        obs = agents.prepare_comm_obs(obs, 0.0)
        msg = None
    else:
        msg = _get_env_msg(env, args, args.n_agents)
    actions = []
    actions_onehot = []
    avail_actions = []
    low_continuous = getattr(args, "low_action_type", "discrete") == "continuous"
    low_action_dim = int(getattr(args, "low_action_dim", args.n_actions))
    for agent_id in range(args.n_agents):
        avail_action = env.get_avail_agent_actions(agent_id)
        action = agents.choose_action(
            obs[agent_id],
            last_action[agent_id],
            agent_id,
            avail_action,
            0.0,
            timestep_cur=step,
            timestep_max=args.n_steps,
            msg=None if msg is None else msg[agent_id],
        )
        if low_continuous:
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            if action.size < low_action_dim:
                action = np.pad(action, (0, low_action_dim - action.size))
            action_onehot = np.clip(action[:low_action_dim], -1.0, 1.0).astype(np.float32)
            actions.append(action_onehot.copy())
        else:
            action_onehot = np.zeros(args.n_actions, dtype=np.float32)
            action_onehot[int(action)] = 1.0
            actions.append(int(action))
        actions_onehot.append(action_onehot)
        avail_actions.append(avail_action)
        last_action[agent_id] = action_onehot
    return raw_obs, actions, avail_actions, actions_onehot


def load_student_corrector(env, args, cli_args):
    if args.alg != "hmappo_cbf_flow" or not cli_args.cbf_flow_use_student:
        return None
    base_env = env.env if hasattr(env, "env") else env
    if not hasattr(base_env, "get_cbf_safety_state"):
        return None
    artifact_dir = (
        cli_args.cbf_flow_artifact_dir
        or os.path.join(args.model_dir, args.alg, args.map, "cbf_flow")
    )
    deploy_path = os.path.join(artifact_dir, "student_deploy.pt")
    fallback_path = os.path.join(artifact_dir, "student.pt")
    model_path = deploy_path if os.path.exists(deploy_path) else fallback_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No distilled student checkpoint found under {artifact_dir}")
    joint_action_dim = int(args.n_agents * int(getattr(args, "low_action_dim", args.n_actions)))
    student = DistilledCorrector(
        safety_state_dim=len(base_env.get_cbf_safety_state()),
        joint_action_dim=joint_action_dim,
        hidden_dim=int(getattr(args, "flow_hidden_dim", 256)),
    )
    device = torch.device(
        f"cuda:{getattr(args, 'gpu_id', 0)}"
        if getattr(args, "cuda", False) and torch.cuda.is_available()
        else "cpu"
    )
    student.load_state_dict(torch.load(model_path, map_location=device))
    student.to(device)
    student.eval()
    print(f"Loaded distilled student corrector: {model_path}")
    return student


def apply_student_corrector(env, args, student, actions):
    if student is None:
        return actions
    base_env = env.env if hasattr(env, "env") else env
    state = torch.as_tensor(
        base_env.get_cbf_safety_state(), dtype=torch.float32, device=next(student.parameters()).device
    ).view(1, -1)
    raw = torch.as_tensor(
        np.asarray(actions, dtype=np.float32).reshape(1, -1),
        dtype=torch.float32,
        device=next(student.parameters()).device,
    )
    with torch.no_grad():
        corrected = student(state, raw).detach().cpu().numpy().reshape(args.n_agents, -1)
    return [corrected[idx].astype(np.float32) for idx in range(args.n_agents)]


def apply_high_level_action_if_needed(env, agents, args, step, current_subgoals):
    level_training = bool(
        getattr(args, "is_level_training", False)
        and hasattr(env, "get_high_level_obs")
        and hasattr(env, "apply_high_level_actions")
    )
    if not level_training:
        return current_subgoals

    meta_period = max(1, int(getattr(args, "hmappo_meta_period", 5)))
    active_agent_mask = _get_active_agent_mask(env, args.n_agents)
    force_update = False
    if (
        current_subgoals is not None
        and getattr(args, "hrl_meta_update_on_subgoal_done", True)
        and hasattr(env, "get_subgoal_success_mask")
    ):
        success = env.get_subgoal_success_mask(current_subgoals)
        force_update = bool(np.any(success * active_agent_mask > 0.0))

    if step % meta_period != 0 and not force_update:
        return current_subgoals

    if hasattr(env, "prepare_high_level_decision"):
        env.prepare_high_level_decision()
    high_obs = env.get_high_level_obs()
    high_avail = (
        env.get_high_level_avail_actions()
        if hasattr(env, "get_high_level_avail_actions")
        else np.ones((args.n_agents, args.high_level_n_actions), dtype=np.float32)
    )
    high_actions = []
    for agent_id in range(args.n_agents):
        if active_agent_mask[agent_id] <= 0.0:
            high_action = np.zeros(args.high_level_n_actions, dtype=np.float32)
        else:
            high_action = agents.choose_high_level_action(
                high_obs[agent_id],
                agent_id,
                high_avail[agent_id],
                0.0,
            )
        high_actions.append(np.asarray(high_action, dtype=np.float32))

    env.apply_high_level_actions(high_actions)
    if hasattr(env, "get_current_subgoals"):
        return env.get_current_subgoals()
    return current_subgoals


def run_episode(env, agents, args, cli_args, episode_idx, student_corrector=None):
    if cli_args.seed is None:
        seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
        print(f"Episode {episode_idx} seed: {seed} (random)")
    else:
        seed = int(cli_args.seed + episode_idx)
        print(f"Episode {episode_idx} seed: {seed}")
    set_episode_seed(seed)
    reset_env(env, seed=seed)
    if hasattr(agents, "reset_episode_state"):
        agents.reset_episode_state()
    if hasattr(agents.policy, "init_hidden"):
        agents.policy.init_hidden(1)
    if hasattr(env, "set_meta_period"):
        env.set_meta_period(getattr(args, "hmappo_meta_period", 5))
    if hasattr(env, "set_hrl_parameters"):
        env.set_hrl_parameters(
            reachable_subgoal_scale=getattr(args, "hrl_reachable_subgoal_scale", None),
            intrinsic_reward_scale=getattr(args, "hrl_intrinsic_reward_scale", None),
            intrinsic_success_bonus=getattr(args, "hrl_intrinsic_success_bonus", None),
            intrinsic_collision_penalty=getattr(
                args, "hrl_intrinsic_collision_penalty", None
            ),
            order_progress_override=getattr(
                args, "hrl_order_progress_override", None
            ),
            energy_margin_reserve_ratio=getattr(
                args, "hrl_energy_margin_reserve_ratio", None
            ),
            charge_queue_enabled=getattr(
                args, "hrl_charge_queue_enabled", None
            ),
            charge_queue_radius=getattr(
                args, "hrl_charge_queue_radius", None
            ),
        )

    max_steps = cli_args.max_steps or args.episode_limit
    terminated = False
    step = 0
    episode_reward = 0.0
    last_action = np.zeros((args.n_agents, args.n_actions), dtype=np.float32)
    info = {
        "battle_won": False,
    }
    current_high_subgoals = None

    frames = []
    save_frame_images = bool(cli_args.save_frames and cli_args.render_mode == "rgb_array")
    xy_frames = (
        []
        if cli_args.render
        and cli_args.render_mode == "rgb_array"
        and cli_args.save_xy
        else None
    )
    render_mode = cli_args.render_mode if (cli_args.render or save_frame_images) else "none"
    render_frame(
        env,
        render_mode,
        frames,
        cli_args.fps,
        xy_frames=xy_frames,
        save_xy=bool(xy_frames is not None),
        save_frames=save_frame_images,
        frame_root=cli_args.frame_root,
        alg=args.alg,
        map_name=args.map,
        episode_idx=episode_idx,
        frame_idx=step,
    )

    while not terminated and step < max_steps:
        current_high_subgoals = apply_high_level_action_if_needed(
            env,
            agents,
            args,
            step,
            current_high_subgoals,
        )
        raw_obs, actions, avail_actions, _ = choose_actions(env, agents, args, last_action, step)
        if student_corrector is not None:
            actions = apply_student_corrector(env, args, student_corrector, actions)
        elif hasattr(agents, "revise_safe_actions"):
            revised_actions = agents.revise_safe_actions(
                observations=raw_obs,
                avail_actions=avail_actions,
                base_actions=actions,
            )
            if revised_actions is not None:
                if getattr(args, "low_action_type", "discrete") == "continuous":
                    actions = [np.asarray(action, dtype=np.float32) for action in revised_actions]
                else:
                    actions = [int(action) for action in revised_actions]
        reward, terminated, info = env.step(actions)
        episode_reward += float(np.asarray(reward, dtype=np.float32).mean())
        step += 1
        render_frame(
            env,
            render_mode,
            frames,
            cli_args.fps,
            xy_frames=xy_frames,
            save_xy=bool(xy_frames is not None),
            save_frames=save_frame_images,
            frame_root=cli_args.frame_root,
            alg=args.alg,
            map_name=args.map,
            episode_idx=episode_idx,
            frame_idx=step,
        )

    win_tag = bool(terminated and info.get("battle_won", False))
    summary = _build_env_summary(env, info, step, win_tag, args.n_agents)
    summary["episode_reward"] = episode_reward

    if render_mode == "rgb_array" and frames:
        output_path = gif_path(cli_args.result_root, args.alg, args.map, episode_idx)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=max(1, int(1000 / cli_args.gif_fps)),
            loop=0,
        )
        print(f"Saved visualization to {output_path}")
        if xy_frames:
            xy_output_path = gif_path(
                cli_args.result_root, args.alg, args.map, episode_idx, suffix="xy"
            )
            xy_frames[0].save(
                xy_output_path,
                save_all=True,
                append_images=xy_frames[1:],
                duration=max(1, int(1000 / cli_args.gif_fps)),
                loop=0,
            )
            print(f"Saved XY visualization to {xy_output_path}")

    if save_frame_images:
        frame_base = os.path.join(
            cli_args.frame_root,
            args.alg,
            args.map,
            f"episode_{episode_idx}",
        )
        print(
            f"Saved per-frame images to {frame_base}/3d and {frame_base}/xy"
        )

    return summary


def main():
    cli_args = parse_args()
    if cli_args.cuda and torch.cuda.is_available():
        torch.cuda.set_device(int(cli_args.gpu_id))
    args = make_base_args(cli_args)
    env = None

    try:
        env = build_env(args, [args.alg])
        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        args.raw_obs_shape = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]
        args.low_action_type = env_info.get("low_action_type", "discrete")
        args.low_action_dim = env_info.get("low_action_dim", args.n_actions)
        args.msg_shape = env_info.get("msg_shape", 0)
        args.high_level_n_actions = env_info.get("high_level_n_actions", 0)
        args.high_level_mode_n_actions = env_info.get("high_level_mode_n_actions", 0)
        args.high_level_obs_shape = env_info.get("high_level_obs_shape", 0)
        args.high_level_state_shape = env_info.get("high_level_state_shape", 0)
        args.low_task_shape = env_info.get("low_task_shape", 0)
        args.max_active_orders = env_info.get(
            "max_active_orders", getattr(args, "uav_max_active_orders", 0)
        )
        args.charge_action_id = env_info.get("charge_action_id", args.max_active_orders)
        args = configure_algorithm_args(args)

        agents = Agents(args, env)
        student_corrector = load_student_corrector(env, args, cli_args)
        print_model_info(args)

        if cli_args.render:
            if cli_args.render_mode == "human":
                print(f"Render mode: human at about {cli_args.fps} FPS.")
            else:
                print(
                    f"Render mode: rgb_array, GIF output in "
                    f"{os.path.abspath(cli_args.result_root)}."
                )
                if cli_args.save_frames:
                    print(
                        "Per-frame 3D and XY images will be saved in "
                        f"{os.path.abspath(cli_args.frame_root)}."
                    )
        else:
            print("Render mode: none.")
            if cli_args.save_frames:
                print(
                    "Per-frame 3D and XY images will be saved in "
                    f"{os.path.abspath(cli_args.frame_root)}."
                )

        summaries = []
        for episode_idx in range(cli_args.episodes):
            summary = run_episode(
                env,
                agents,
                args,
                cli_args,
                episode_idx,
                student_corrector=student_corrector,
            )
            summaries.append(summary)
            print(
                f"Episode {episode_idx}: reward={summary['episode_reward']:.3f}, "
                f"win={int(bool(summary['win_tag']))}, steps={int(summary['step'])}, "
                f"collisions={float(summary.get('collision_count', 0.0)):.1f}, "
                f"obstacle_collisions={float(summary.get('obstacle_collision_count', 0.0)):.1f}, "
                f"agent_collisions={float(summary.get('agent_collision_count', 0.0)):.1f}"
            )

        avg_reward = float(np.mean([summary["episode_reward"] for summary in summaries]))
        avg_win_rate = float(np.mean([summary["win_tag"] for summary in summaries]))
        avg_steps = float(np.mean([summary["step"] for summary in summaries]))
        avg_collision = float(
            np.mean([summary.get("collision_count", 0.0) for summary in summaries])
        )
        avg_obstacle_collision = float(
            np.mean(
                [summary.get("obstacle_collision_count", 0.0) for summary in summaries]
            )
        )
        avg_agent_collision = float(
            np.mean(
                [summary.get("agent_collision_count", 0.0) for summary in summaries]
            )
        )
        print(
            f"Average over {len(summaries)} episode(s): "
            f"reward={avg_reward:.3f}, win_rate={avg_win_rate:.3f}, steps={avg_steps:.1f}, "
            f"collisions={avg_collision:.1f}, obstacle_collisions={avg_obstacle_collision:.1f}, "
            f"agent_collisions={avg_agent_collision:.1f}"
        )
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
