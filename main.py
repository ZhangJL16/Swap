import matplotlib
matplotlib.use("Agg")

from runner import Runner
from common.arguments import (
    get_common_args,
    get_coma_args,
    get_mixer_args,
    get_centralv_args,
    get_reinforce_args,
    get_commnet_args,
    get_g2anet_args,
    get_RGM_args,
    get_ippo_args,
    get_mappo_args,
    get_mappo_lagrangian_args,
    get_macpo_args,
    lr_adjust,
)
from common.rollout import SMAC_MAPS, smac_penalty_enabled
from common.seeding import seed_everything
import matplotlib.pyplot as plt
import os
import torch
from datetime import datetime
from contextlib import contextmanager

plt.style.use("bmh")
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.unicode_minus"] = True
plt.rcParams["text.color"] = "black"
plt.set_cmap("jet")
colors = ["blue", "orange", "red", "forestgreen", "darkviolet", "black"]


def _charging_capacity_arg(args):
    capacity = getattr(args, "uav_charging_capacity", None)
    if capacity is not None:
        return int(capacity)
    return 2


def _charging_station_count_arg(args):
    station_count = getattr(args, "uav_charging_station_count", None)
    if station_count is not None:
        return int(station_count)
    return max(1, (int(getattr(args, "uav_n_agents", 4)) + 1) // 2)


@contextmanager
def _suppress_stdio(enabled=True):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        stdout_fd = os.dup(1)
        stderr_fd = os.dup(2)
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)


class _SilentSMACEnv:
    def __init__(self, env, enabled=True):
        self._env = env
        self._enabled = bool(enabled)

    def __getattr__(self, item):
        return getattr(self._env, item)

    def reset(self, *args, **kwargs):
        with _suppress_stdio(self._enabled):
            return self._env.reset(*args, **kwargs)

    def close(self):
        with _suppress_stdio(self._enabled):
            return self._env.close()


def _configure_sc2_path(args):
    if args.map not in SMAC_MAPS:
        return

    if args.sc2_path:
        os.environ["SC2PATH"] = args.sc2_path

    sc2_root = os.environ.get("SC2PATH", "")
    if not sc2_root and os.name == "nt":
        sc2_root = r"C:/Program Files (x86)/StarCraft II"

    if not sc2_root:
        return

    versions_dir = os.path.join(sc2_root, "Versions")
    if not os.path.isdir(versions_dir):
        raise FileNotFoundError(
            "SMAC map requires a valid StarCraft II installation. "
            f"Current SC2 root: '{sc2_root}', but '{versions_dir}' was not found. "
            "Pass --sc2_path \"D:/Program Files (x86)/StarCraft II\" "
            "or set environment variable SC2PATH."
        )


def build_env(args, algs):
    def local_episode_limit(default=400):
        value = getattr(args, "episode_limit", None)
        return default if value is None else int(value)

    if args.map in {"simple_speaker_listener", "simple_speaker_listener_v4", "MPE2SpeakerListener"}:
        try:
            from envs.MPE2Env import MPE2SimpleSpeakerListenerWrapper
        except ImportError as exc:
            raise ImportError(
                "Map 'simple_speaker_listener' requires local module envs.MPE2Env."
            ) from exc
        return MPE2SimpleSpeakerListenerWrapper()

    if args.map == "UAV2D":
        try:
            from envs.UAVEnv import UAVEnvDiscreteWrapper
        except ImportError as exc:
            raise ImportError(
                "Map 'UAV2D' requires local module envs.UAVEnv."
            ) from exc
        return UAVEnvDiscreteWrapper(
            dim_actions=2,
            num_hunters=int(getattr(args, "uav_n_agents", 4)),
            episode_limit=local_episode_limit(),
        )

    if args.map == "UAV3D":
        try:
            from envs.UAVEnv import UAVEnvDiscreteWrapper
        except ImportError as exc:
            raise ImportError(
                "Map 'UAV3D' requires local module envs.UAVEnv."
            ) from exc
        return UAVEnvDiscreteWrapper(
            dim_actions=3,
            num_hunters=int(getattr(args, "uav_n_agents", 4)),
            episode_limit=local_episode_limit(),
        )

    if args.map in {"UAVDelivery", "UAVDelivery2D", "UAVDelivery3D"}:
        try:
            from envs.UAVDelivery import UAVEnvDiscreteWrapper
        except ImportError as exc:
            raise ImportError(
                "Map 'UAVDelivery' requires local module envs.UAVDelivery."
            ) from exc
        dim_actions = 3 if args.map == "UAVDelivery3D" else 2
        return UAVEnvDiscreteWrapper(
            dim_actions=dim_actions,
            num_hunters=int(getattr(args, "uav_n_agents", 4)),
            episode_limit=local_episode_limit(),
            total_orders=int(getattr(args, "uav_total_orders", 8)),
            max_active_orders=int(getattr(args, "uav_max_active_orders", 4)),
            pickup_reward=float(getattr(args, "uav_pickup_reward", 3.0)),
            delivery_reward=float(getattr(args, "uav_delivery_reward", 8.0)),
        )

    if args.map in {
        "UAVEnergyDelivery",
        "UAVEnergyDelivery2D",
        "UAVEnergyDelivery3D",
    }:
        try:
            from envs.UAVEnergyDelivery import UAVEnvDiscreteWrapper
        except ImportError as exc:
            raise ImportError(
                "Map 'UAVEnergyDelivery' requires local module envs.UAVEnergyDelivery."
            ) from exc
        dim_actions = 3 if args.map == "UAVEnergyDelivery3D" else 2
        return UAVEnvDiscreteWrapper(
            dim_actions=dim_actions,
            num_hunters=int(getattr(args, "uav_n_agents", 4)),
            episode_limit=local_episode_limit(),
            total_orders=int(getattr(args, "uav_total_orders", 8)),
            max_active_orders=int(getattr(args, "uav_max_active_orders", 4)),
            pickup_reward=float(getattr(args, "uav_pickup_reward", 3.0)),
            delivery_reward=float(getattr(args, "uav_delivery_reward", 8.0)),
            initial_energy=float(getattr(args, "uav_initial_energy", 100.0)),
            energy_decay_per_step=getattr(args, "uav_energy_decay", None),
            energy_depletion_fraction=float(
                getattr(args, "uav_energy_depletion_fraction", 0.5)
            ),
            charging_capacity=_charging_capacity_arg(args),
            charging_station_count=_charging_station_count_arg(args),
            charging_radius=float(getattr(args, "uav_charging_radius", 0.18)),
            charging_rate=getattr(args, "uav_charging_rate", None),
        )

    if args.map in {"UAVEncircle", "UAVencircle"}:
        try:
            from envs.UAVEncircle import UAVEnvDiscreteWrapper
        except ImportError as exc:
            raise ImportError(
                "Map 'UAVEncircle' requires local module envs.UAVEncircle."
            ) from exc
        return UAVEnvDiscreteWrapper(
            dim_actions=3,
            num_hunters=int(getattr(args, "uav_n_agents", 4)),
            episode_limit=local_episode_limit(),
        )

    if args.map == "Basic2P":
        try:
            from envs.maze.Basic2P import Basic2P
        except ImportError as exc:
            raise ImportError(
                "Map 'Basic2P' requires local module envs.maze.Basic2P."
            ) from exc
        return Basic2P(symmetric=True, args=args)

    if args.map == "IoV":
        try:
            from envs.crypt.IoV import IoV
        except ImportError as exc:
            raise ImportError(
                "Map 'IoV' requires local module envs.crypt.IoV."
            ) from exc
        return IoV(args)

    if args.map == "stego":
        try:
            from envs.stego.imagestego import ImageStegoEnv
        except ImportError as exc:
            raise ImportError(
                "Map 'stego' requires local module envs.stego.imagestego."
            ) from exc
        return ImageStegoEnv(args)

    if args.map in SMAC_MAPS:
        _configure_sc2_path(args)
        try:
            from smac.env import StarCraft2Env
        except ImportError as exc:
            raise ImportError(
                "SMAC map selected, but smac is not installed in this Python environment."
            ) from exc

        replay_dir = args.replay_dir
        if replay_dir != "":
            exp = ""
            for alg in algs:
                exp += alg + "_"
            exp = exp[:-1] + "/"
            replay_dir = os.getcwd() + "/replay/" + exp
            if not os.path.exists(replay_dir):
                os.makedirs(replay_dir)
            args.replay_dir = replay_dir

        with _suppress_stdio(not bool(args.debug)):
            base_env = StarCraft2Env(
                map_name=args.map,
                step_mul=args.step_mul,
                difficulty=args.difficulty,
                game_version=args.game_version,
                replay_dir=args.replay_dir,
                debug=args.debug,
            )
        base_env = _SilentSMACEnv(base_env, enabled=not bool(args.debug))
        if smac_penalty_enabled(args):
            try:
                from envs.SMACSafeEnv import SMACSafetyWrapper
            except ImportError as exc:
                raise ImportError(
                    "SMAC safety wrapper requires local module envs.SMACSafeEnv."
                ) from exc
            return SMACSafetyWrapper(
                base_env,
                risk_threshold=float(getattr(args, "guard_risk_threshold", 0.1)),
            )
        return base_env

    raise ValueError(
        "Unknown map '{}'. Supported SMAC maps: {}".format(
            args.map, ", ".join(SMAC_MAPS)
        )
    )

if __name__ == "__main__":  
    # suffix: _Comm 进行通信 _reshape 短期
    # prefix : g 长期
    algs = ['mappo_reshape',  
         # 'iql', 'iql_reshape_Comm', 'iql_Comm', 'mappo',  'mappo_reshape', 'RGMComm', 
    ]  # ]'gmix_reshape', 'gmix_reshape_Comm' 'qmix' 'qmix_reshape_Comm', 'qmix', 'vdn' 'qmix_Comm',
    # algs = ['vdn', 'vdn_Comm', 'vdn_reshape', 'vdn_reshape_Comm' # "iql_Comm"， 'gmix_reshape'
    args = get_common_args()
    args.now = datetime.now().strftime("%m%d_%H%M%S")
    seed_everything(
        args.seed,
        deterministic_torch=bool(getattr(args, "deterministic_torch", True)),
    )
    if args.cuda and torch.cuda.is_available():
        torch.cuda.set_device(int(getattr(args, "gpu_id", 0)))
    
    # 支持两种模式：
    # 1. 单算法模式：使用 --alg 参数（如 python main.py --alg mappo --map Basic2P）
    # 2. 多算法对比模式：使用 --alg_list 参数（如 python main.py --alg_list "mappo,qmix,vdn"）
    if args.alg_list:
        algs = [alg.strip() for alg in args.alg_list.split(',')]
    else:
        algs = [args.alg]  # 单算法模式

    for i, alg in enumerate(algs):
        args.alg = alg
        seed_everything(
            args.seed,
            deterministic_torch=bool(getattr(args, "deterministic_torch", True)),
        )
        print(f"algo: {args.alg}")

        # lr_adjust(args, alg)

        env = build_env(args, algs)

        env_info = env.get_env_info()

        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        args.raw_obs_shape = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]
        args.msg_shape = env_info.get("msg_shape", 0)

        # import pdb; pdb.set_trace()

        if args.alg.find("coma") > -1:
            args = get_coma_args(args)
        elif args.alg.find("central_v") > -1:
            args = get_centralv_args(args)
        elif args.alg.find("reinforce") > -1:
            args = get_reinforce_args(args)
        elif args.alg.find("ippo") > -1:
            args = get_ippo_args(args)
        elif args.alg.find("mappo_lagrangian") > -1:
            args = get_mappo_lagrangian_args(args)
        elif args.alg.find("mappo") > -1:
            args = get_mappo_args(args)
        elif args.alg.find("macpo") > -1:
            args = get_macpo_args(args)
        elif args.alg.lower().find("rgmcomm") > -1:
            args = get_RGM_args(args)
        else:
            args = get_mixer_args(args)
        # CommAgent
        if args.alg.find("commnet") > -1:
            args = get_commnet_args(args)
        if args.alg.find("g2anet") > -1:
            args = get_g2anet_args(args)

        if alg.lower().find("comm") > -1 and alg.lower().find("rgmcomm") < 0:
            args.msg_shape = min(
                int(getattr(args, "comm_max_keep_dim", args.raw_obs_shape)),
                int(args.raw_obs_shape),
            )
            args.obs_shape += args.msg_shape * (args.n_agents - 1)
            args.state_shape = args.obs_shape * args.n_agents

        # import pdb; pdb.set_trace()

        runner = Runner(env, args)
        if alg.lower().find("rgmcomm") == -1:
            runner.run(i)
        else:
            runner.runRGM()

        env.close()

    # plt_multi_alg(args, algs)
