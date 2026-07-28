import argparse
import os

"""
Here are the param for the training

"""


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def get_common_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--difficulty', type=str, default='7', help='the difficulty of the game')
    parser.add_argument('--game_version', type=str, default='latest', help='the version of the game')
    parser.add_argument('--map', type=str, default='Basic2P', help='the map of the game')
    parser.add_argument('--sc2_path', type=str, default='/home/ubuntu/桌面/cc/StarCraftII', help='StarCraft II installation root path, e.g. "C:/Program Files (x86)/StarCraft II"')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--eval_seed', type=int, default=None, help='base seed for fixed evaluation episodes; defaults to seed + 100000')
    parser.add_argument('--episode_seed_stride', type=int, default=1, help='stride between deterministic episode seeds')
    parser.add_argument('--deterministic_torch', dest='deterministic_torch', action='store_true', default=True, help='use deterministic PyTorch settings when possible')
    parser.add_argument('--no_deterministic_torch', dest='deterministic_torch', action='store_false', help='disable deterministic PyTorch settings')
    parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
    parser.add_argument('--replay_dir', type=str, default='./replay', help='absolute path to save the replay') # ./replay
    parser.add_argument('--debug', type=_str2bool, default=False, help='smac show infos')
    parser.add_argument('--uav_n_agents', type=int, default=4, help='number of UAV agents for UAV2D/UAV3D/UAVEncircle/UAVDelivery environments')
    parser.add_argument('--episode_limit', type=int, default=400, help='episode length for local UAV environments')
    parser.add_argument('--uav_total_orders', type=int, default=16, help='total delivery orders per UAVDelivery episode')
    parser.add_argument('--uav_max_active_orders', type=int, default=8, help='maximum simultaneously active delivery orders in UAVDelivery')
    parser.add_argument('--uav_pickup_reward', type=float, default=3.0, help='reward for reaching a UAVDelivery pickup point')
    parser.add_argument('--uav_delivery_reward', type=float, default=8.0, help='reward for reaching a UAVDelivery dropoff point')
    parser.add_argument('--uav_initial_energy', type=float, default=100.0, help='initial energy for UAVEnergyDelivery agents')
    parser.add_argument('--uav_energy_decay', type=float, default=None, help='energy consumed per UAVEnergyDelivery step; defaults to depletion at half episode length')
    parser.add_argument('--uav_energy_depletion_fraction', type=float, default=0.5, help='fraction of episode length when full energy should be depleted')
    parser.add_argument('--uav_energy_model', type=str, default='fixed', choices=['fixed', 'dynamic'], help='UAVEnergyDelivery energy model: fixed per step or velocity/acceleration/payload dependent')
    parser.add_argument('--uav_energy_idle_coef', type=float, default=None, help='dynamic energy base power coefficient; defaults to uav_energy_decay / env time_step')
    parser.add_argument('--uav_energy_speed_coef', type=float, default=2.0, help='dynamic energy coefficient for squared UAV speed')
    parser.add_argument('--uav_energy_accel_coef', type=float, default=30.0, help='dynamic energy coefficient for squared UAV acceleration')
    parser.add_argument('--uav_energy_payload_coef', type=float, default=0.1, help='dynamic energy additive power coefficient while carrying an order')
    parser.add_argument('--uav_charging_station_count', type=int, default=None, help='number of charging stations; defaults to ceil(uav_n_agents / 2)')
    parser.add_argument('--uav_charging_capacity', type=int, default=2, help='maximum UAVs charging at each station in one step')
    parser.add_argument('--uav_charging_radius', type=float, default=0.18, help='distance threshold for UAVEnergyDelivery charging station')
    parser.add_argument('--uav_charging_rate', type=float, default=None, help='energy restored per charging step; defaults to 4x energy decay per step')
    parser.add_argument('--hmappo_meta_period', type=int, default=5, help='number of low-level steps between hierarchical high-level decisions')
    parser.add_argument('--high_lr_actor', type=float, default=None, help='actor learning rate for hierarchical high-level policy')
    parser.add_argument('--high_lr_critic', type=float, default=None, help='critic learning rate for hierarchical high-level policy')
    parser.add_argument('--high_actor_hidden_dim', type=int, default=None, help='hidden dimension for hierarchical high-level actor')
    parser.add_argument('--high_critic_hidden_dim', type=int, default=None, help='hidden dimension for hierarchical high-level critic')
    parser.add_argument('--hrl_use_intrinsic_reward', type=_str2bool, default=False, help='legacy compatibility flag; low-level HMAPPO now uses environment rewards')
    parser.add_argument('--hrl_intrinsic_reward_scale', type=float, default=1.0, help='legacy compatibility value; no intrinsic reward is applied')
    parser.add_argument('--hrl_intrinsic_success_bonus', type=float, default=1.0, help='legacy compatibility value; no intrinsic success bonus is applied')
    parser.add_argument('--hrl_high_goal_style', type=str, default='line', choices=['line_lateral', 'target_relative', 'free_relative', 'line'], help='high-level continuous action semantics: legacy line progress, line progress plus bounded lateral residual, target-conditioned HIRO goal, or free relative dx/dy goal')
    parser.add_argument('--hrl_high_mode_policy', type=str, default='hybrid', choices=['hybrid', 'continuous'], help='high-level mode policy: hybrid uses categorical charge/order, continuous uses a Gaussian scalar thresholded by charge_mode_fraction')
    parser.add_argument('--hrl_high_lateral_scale', type=float, default=0.35, help='maximum lateral residual as a fraction of the reachable high-level subgoal radius for line_lateral/target_relative goals')
    parser.add_argument('--hrl_order_progress_override', type=float, default=None, help='legacy line-goal only: override high-level progress scalar for order-mode subgoals; disabled when unset')
    parser.add_argument('--hrl_depleted_terminal_cost_coef', type=float, default=0.0, help='terminal high-level cost coefficient applied per agent that ends an episode with depleted battery')
    parser.add_argument('--hrl_reachable_subgoal_scale', type=float, default=1.0, help='scale applied to the reachable local subgoal radius')
    parser.add_argument('--hrl_oracle_high_level', type=_str2bool, default=False, help='use oracle order-seeking high-level actions for low-level pretraining')
    parser.add_argument('--hmappo_freeze_high_level', type=_str2bool, default=False, help='skip high-level policy updates in HMAPPO')
    parser.add_argument('--hmappo_freeze_low_level', type=_str2bool, default=False, help='skip low-level policy updates in HMAPPO')
    parser.add_argument('--hmappo_pretrained_low_model_dir', type=str, default='', help='directory containing pretrained HMAPPO low-level actor/critic checkpoints')
    parser.add_argument('--hrl_meta_update_on_subgoal_done', type=_str2bool, default=False, help='legacy compatibility flag; high-level target refresh follows hmappo_meta_period')
    parser.add_argument('--hrl_safe_action_guard_enabled', type=_str2bool, default=False, help='enable a SCOPE-style deterministic low-level safety action replacement for local UAV HMAPPO')
    parser.add_argument('--hrl_safe_action_guard_margin', type=float, default=0.04, help='extra collision prediction margin used by the low-level safety action replacement')
    parser.add_argument('--hrl_safe_action_guard_horizon', type=int, default=4, help='number of future low-level steps checked by the deterministic safety action replacement')
    parser.add_argument('--hrl_energy_margin_reserve_ratio', type=float, default=0.05, help='reserve ratio used only for selected-order energy margin observation/diagnostics')
    parser.add_argument('--hrl_auction_enabled', type=_str2bool, default=True, help='enable auction-based order pre-assignment in UAVEnergyDeliveryLevel')
    parser.add_argument('--hrl_charge_energy_threshold', type=float, default=0.35, help='energy ratio below which agents with no current order fall back to charging')
    parser.add_argument('--hrl_charge_release_threshold', type=float, default=0.65, help='energy ratio at which an ongoing charge option is released')
    parser.add_argument('--hrl_charge_queue_enabled', type=_str2bool, default=False, help='spread charging subgoals across capacity-aware dock and waiting slots')
    parser.add_argument('--hrl_charge_queue_radius', type=float, default=0.24, help='radius of the charging waiting ring when charge queue is enabled')
    parser.add_argument('--hrl_charge_mode_fraction', type=float, default=0.5, help='fraction of the high-level mode interval reserved for charging; 0.25 means charge:order = 1:3')
    parser.add_argument('--hrl_charge_dense_reward_scale', type=float, default=0.0, help='scale for dense goal-shaping rewards when the current high-level target is charging')
    parser.add_argument('--training_phase', type=str, default='low_cbf', choices=['low_cbf', 'flow', 'distill', 'energy', 'high_q'], help='training phase for hmappo_cbf_flow')
    parser.add_argument('--agent_entry_interval', type=int, default=1, help='hmappo_cbf_flow: number of env steps between UAV entries')
    parser.add_argument('--energy_gamma', type=float, default=0.99, help='navigation energy TD discount')
    parser.add_argument('--energy_lr', type=float, default=3e-4, help='navigation energy learning rate')
    parser.add_argument('--energy_reserve', type=float, default=0.0, help='energy reserve used by high-level feasibility filtering')
    parser.add_argument('--energy_buffer_size', type=int, default=100000, help='navigation energy replay buffer size')
    parser.add_argument('--energy_batch_size', type=int, default=256, help='navigation energy replay batch size')
    parser.add_argument('--order_q_lr', type=float, default=3e-4, help='high-level order Q learning rate')
    parser.add_argument('--order_q_gamma', type=float, default=0.99, help='high-level SMDP order Q discount')
    parser.add_argument('--order_q_buffer_size', type=int, default=100000, help='high-level order Q replay buffer size')
    parser.add_argument('--order_q_batch_size', type=int, default=256, help='high-level order Q replay batch size')
    parser.add_argument('--order_max_duration', type=int, default=120, help='maximum low-level steps for a high-level option')
    parser.add_argument('--cbf_alpha1', type=float, default=1.0, help='HOCBF alpha1')
    parser.add_argument('--cbf_alpha2', type=float, default=1.0, help='HOCBF alpha2')
    parser.add_argument('--cbf_teacher_margin', type=float, default=0.0, help='CBF-QP teacher margin')
    parser.add_argument('--flow_lr', type=float, default=3e-4, help='constrained flow learning rate')
    parser.add_argument('--flow_hidden_dim', type=int, default=256, help='constrained flow MLP hidden dimension')
    parser.add_argument('--flow_train_steps', type=int, default=8, help='Euler steps used during flow training')
    parser.add_argument('--flow_eval_steps', type=int, default=8, help='Euler steps used during flow evaluation')
    parser.add_argument('--flow_cbf_coef', type=float, default=1.0, help='flow CBF loss coefficient')
    parser.add_argument('--flow_endpoint_coef', type=float, default=1.0, help='flow endpoint loss coefficient')
    parser.add_argument('--flow_terminal_coef', type=float, default=1.0, help='flow terminal CBF loss coefficient')
    parser.add_argument('--flow_margin', type=float, default=0.0, help='flow safety margin')
    parser.add_argument('--flow_kappa', type=float, default=1.0, help='flow CBF margin-rate coefficient')
    parser.add_argument('--student_lr', type=float, default=3e-4, help='distilled corrector learning rate')
    parser.add_argument('--student_teacher_coef', type=float, default=1.0, help='student teacher-action loss coefficient')
    parser.add_argument('--student_safety_coef', type=float, default=1.0, help='student safety loss coefficient')
    parser.add_argument('--student_margin', type=float, default=0.0, help='student safety margin')
    parser.add_argument('--actor_correct_align_coef', type=float, default=0.0, help='low-level actor alignment coefficient toward CBF-corrected actions')
    parser.add_argument('--cbf_flow_artifact_dir', type=str, default='', help='hmappo_cbf_flow artifact directory; defaults to model_dir/alg/map/cbf_flow')
    parser.add_argument('--cbf_flow_load_artifact_dir', type=str, default='', help='directory to load hmappo_cbf_flow staged artifacts from')
    parser.add_argument('--cbf_flow_student_mse_threshold', type=float, default=0.02, help='maximum student teacher MSE allowed for deployment export')
    parser.add_argument('--cbf_flow_export_student', type=_str2bool, default=False, help='export student deployment checkpoint only after validation passes')
    parser.add_argument(
        '--experiment_device',
        type=str,
        default=os.environ.get('MARL_EXPERIMENT_DEVICE', 'dorm'),
        help='device label written to the UAVDelivery experiment summary CSV',
    )

    # The alternative algorithms are vdn, coma, central_v, qmix, qtran_base,
    # qtran_alt, reinforce, coma+commnet, central_v+commnet, reinforce+commnet，
    # coma+g2anet, central_v+g2anet, reinforce+g2anet, maven, mappo, hmappo, RGMComm
    parser.add_argument('--alg', type=str, default='rgmcomm', help='the algorithm to train the agent')
    parser.add_argument('--alg_list', type=str, default='', help='comma-separated list of algorithms to compare (e.g., "mappo,qmix,vdn")')

    # time slot = n_steps / evaluate_cycle
    parser.add_argument('--n_steps', type=int, default=600000, help='total time steps')  # 4000000
    parser.add_argument('--time_steps', type=int, default=None, help='total time steps for RGMComm/MADDPG; defaults to n_steps')
    parser.add_argument('--evaluate_rate', type=int, default=1000, help='evaluation interval for RGMComm/MADDPG/MATD3')
    parser.add_argument('--evaluate_episode_len', type=int, default=100, help='evaluation episode length for RGMComm/MADDPG/MATD3')
    parser.add_argument('--sample_rate', type=int, default=2000, help='Q-sample interval for RGMComm/MADDPG/MATD3 diagnostics')
    parser.add_argument('--sample_start', type=int, default=1000, help='first Q-sample step for RGMComm/MADDPG/MATD3 diagnostics')
    parser.add_argument('--buffer_size', type=int, default=int(5e5), help='replay buffer size for RGMComm/MADDPG/MATD3')
    parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
    parser.add_argument('--save_rate', type=int, default=2000, help='model save interval for RGMComm/MADDPG/MATD3')
    parser.add_argument('--lr_actor', type=float, default=1e-4, help='actor learning rate for RGMComm/MADDPG/MATD3')
    parser.add_argument('--lr_critic', type=float, default=1e-3, help='critic learning rate for RGMComm/MADDPG/MATD3')
    parser.add_argument('--noise_rate', type=float, default=0.1, help='exploration noise rate for RGMComm/MADDPG/MATD3')
    parser.add_argument('--tau', type=float, default=0.01, help='target network soft update coefficient')
    parser.add_argument('--matd3_policy_delay', type=int, default=2, help='critic updates per delayed MATD3 actor update')
    parser.add_argument('--matd3_target_noise', type=float, default=0.2, help='MATD3 target policy smoothing noise scale')
    parser.add_argument('--matd3_target_noise_clip', type=float, default=0.5, help='MATD3 target policy smoothing noise clip scale')
    parser.add_argument('--n_episodes', type=int, default=1, help='the number of episodes before once training')
    # 需要大于一个 episode 的长度
    parser.add_argument('--evaluate_cycle', type=int, default=20, help='how often to evaluate the model') # 5000
    # 每个 time slot 是多少次的平均
    parser.add_argument('--evaluate_epoch', type=int, default=1, help='number of the epoch to evaluate the agent') # 32
    parser.add_argument('--smac_parallel_envs', type=int, default=1, help='number of parallel SMAC rollout workers for qmix/vdn series')

    parser.add_argument('--last_action', type=_str2bool, default=False,
                        help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=_str2bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')
    parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
    parser.add_argument('--load_model', type=_str2bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--evaluate', type=_str2bool, default=False, help='whether to evaluate the model')
    parser.add_argument('--cuda', type=_str2bool, default=True, help='whether to use the GPU')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id to use when CUDA is enabled')
    # parser.add_argument('--record', type=bool, default=True, help='whether to save the exp (fig and numpy)')
    parser.add_argument('--last_reward', type=_str2bool, default=False, help='draw last reward or not')
    # G-network related
    parser.add_argument('--distributed', type=_str2bool, default=True,
                        help='equip every q network with a guide (mix after fix)')
    parser.add_argument('--guide_mix_network_type', type=str, default='vdn', help='use vdn or qmix to sum G values')
    # communication related
    parser.add_argument('--comm', type=_str2bool, default=False, help='whether to communicate')
    parser.add_argument('--msg_size', type=int, default=3, help='size of message')
    parser.add_argument('--aoi_threshold', type=float, default=0.25, help='AoI freshness threshold in seconds')
    parser.add_argument('--aoi_min_weight', type=float, default=0.2, help='minimum retained weight for stale message dimensions')
    parser.add_argument('--aoi_stale_decay', type=float, default=0.5, help='multiplicative decay for stale message dimensions')
    parser.add_argument('--comm_aoi_penalty', type=float, default=0.1, help='AoI penalty applied to communication replay reward')
    parser.add_argument('--comm_warning_penalty', type=float, default=0.2, help='warning penalty applied to communication replay reward')
    parser.add_argument('--comm_warning_threshold', type=float, default=0.1, help='warning threshold for routing communication samples into the risky replay buffer')
    parser.add_argument('--comm_fresh_bonus', type=float, default=0.05, help='freshness bonus applied to communication replay reward')
    parser.add_argument('--comm_cost_penalty', type=float, default=0.1, help='communication cost penalty proportional to the transmitted share ratio')
    parser.add_argument('--comm_effect_bonus', type=float, default=0.1, help='bonus for the effective received message ratio')
    parser.add_argument('--comm_max_keep_dim', type=int, default=8, help='maximum number of observation dimensions that can be shared by the communication module')
    parser.add_argument('--comm_lr', type=float, default=None, help='override learning rate for the communication DQN/C-network')
    parser.add_argument('--safety_lr', type=float, default=None, help='override learning rate for the safety network')
    parser.add_argument('--safety_beta', type=float, default=None, help='override long-term risk weight in short+beta*long')
    parser.add_argument('--warning_penalty_weight', type=float, default=None, help='override warning-signal penalty weight used in reward shaping/logging')
    parser.add_argument('--run_script', type=str, default=os.environ.get('MARL_RUN_SCRIPT', ''), help='training script path written to experiment summary CSV')
    parser.add_argument('--run_command', type=str, default=os.environ.get('MARL_RUN_COMMAND', ''), help='training command written to experiment summary CSV')
    parser.add_argument('--experiment_log_csv', type=str, default=os.environ.get('MARL_EXPERIMENT_LOG_CSV', ''), help='UAV delivery experiment summary CSV path; defaults to train_logs/uav_delivery_experiments.csv')
    # timestamp
    parser.add_argument('--now', type=str, default='', help='timestamp for mat logging')
    args = parser.parse_args()
    if args.eval_seed is None:
        args.eval_seed = args.seed + 100000
    return args


# arguments of coma
def get_coma_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'episode'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # how often to save the model
    args.save_cycle = 500

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of vdn、 qmix、 qtran
def get_mixer_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.mixing_embed_dim = 32
    args.hypernet_layers = 2
    args.hypernet_embed = 64
    args.two_hyper_layers = True
    args.hyper_hidden_dim = 64
    args.qtran_hidden_dim = 64
    args.lr = 1e-4

    # epsilon greedy
    args.epsilon = 1.0
    args.min_epsilon = 0.05
    anneal_steps = 50000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_time = anneal_steps
    args.epsilon_anneal_scale = 'step'
    args.last_action = True
    args.reuse_network = True

    # the number of gradient updates after each collected episode batch
    args.train_steps = 1

    # experience replay
    args.batch_size = 32
    args.buffer_size = int(5e3)

    # how often to save the model
    args.save_cycle = args.n_steps // 4 #5000

    # how often to update the target_net
    args.target_update_cycle = 200
    args.target_update_interval = args.target_update_cycle

    # QTRAN lambda
    args.lambda_opt = 1
    args.lambda_nopt = 1

    # prevent gradient explosion
    args.grad_norm_clip = 10
    args.optim_alpha = 0.99
    args.optim_eps = 1e-5
    args.double_q = True

    # MAVEN
    args.noise_dim = 16
    args.lambda_mi = 0.001
    args.lambda_ql = 1
    args.entropy_coefficient = 0.001
    return args


# arguments of central_v
def get_centralv_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'episode'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of central_v
def get_reinforce_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'episode'

    # how often to save the model
    args.save_cycle = 5000

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of coma+commnet
def get_commnet_args(args):
    if args.map == '3m':
        args.k = 2
    else:
        args.k = 3
    return args


def get_g2anet_args(args):
    args.attention_dim = 32
    args.hard = True
    return args

def get_mappo_args(args):
    """MAPPO 算法的超参数"""
    args.rnn_hidden_dim = 64
    args.actor_hidden_dim = 128
    args.critic_hidden_dim = 128
    args.critic_dim = 128
    args.lr_actor = 3e-4
    args.lr_critic = 3e-4
    args.gamma = 0.95

    # keep the existing epsilon field for logging compatibility; MAPPO does not use it for action selection
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.05
    args.epsilon_anneal_scale = 'episode'

    args.clip_param = 0.2
    args.ppo_epoch = 10
    args.entropy_coef = 1e-3
    args.gae_lambda = 0.95
    args.batch_size = 64
    args.high_lr_actor = (
        args.lr_actor
        if getattr(args, "high_lr_actor", None) is None
        else args.high_lr_actor
    )
    args.high_lr_critic = (
        args.lr_critic
        if getattr(args, "high_lr_critic", None) is None
        else args.high_lr_critic
    )
    args.high_actor_hidden_dim = (
        args.actor_hidden_dim
        if getattr(args, "high_actor_hidden_dim", None) is None
        else args.high_actor_hidden_dim
    )
    args.high_critic_hidden_dim = (
        args.critic_hidden_dim
        if getattr(args, "high_critic_hidden_dim", None) is None
        else args.high_critic_hidden_dim
    )

    # approximate the baseline rollout length (2048) using episode-based collection
    args.n_episodes = max(1, (2048 + args.episode_limit - 1) // args.episode_limit)

    args.save_cycle = 1000
    args.grad_norm_clip = 40
    args.safety_hidden_dim = 128
    args.safety_lr = 3e-4 if args.safety_lr is None else args.safety_lr
    args.safety_gamma = 0.95
    args.safety_beta = 0.8 if args.safety_beta is None else args.safety_beta
    args.safety_target_update_cycle = 200
    args.guard_risk_scale = 1.0
    args.guard_risk_threshold = 0.1
    args.guard_warmup_steps = 10
    args.guard_replace_margin = 0.01

    return args


def get_ippo_args(args):
    """IPPO uses PPO updates with independent local-observation critics."""
    return get_mappo_args(args)


def get_macpo_args(args):
    """MACPO 算法的超参数"""
    args = get_mappo_args(args)

    # cost / constraint related
    args.lr_cost_critic = 5e-4
    args.cost_limit = 0.1
    args.cost_coef = 1.0
    args.lambda_lr = 5e-2
    args.lambda_init = 0.0

    return args


def get_mappo_lagrangian_args(args):
    """MAPPO-Lagrangian hyperparameters."""
    args = get_mappo_args(args)
    args.lr_cost_critic = 5e-4
    args.cost_limit = 0.1
    args.cost_coef = 1.0
    args.lambda_lr = 5e-2
    args.lambda_init = 0.0
    return args

def get_RGM_args(args):
    # 直接给 args 添加 RGM 相关的属性（动态扩展）
    args.scenario_name = getattr(args, 'scenario_name', 'simple_tag_6')  # 可以从外部传，或设默认值
    args.max_episode_len = getattr(args, 'max_episode_len', 100)
    args.time_steps = (
        getattr(args, 'n_steps', 200001)
        if getattr(args, 'time_steps', None) is None
        else int(args.time_steps)
    )
    args.num_adversaries = getattr(args, 'num_adversaries', 1)
    
    args.lr_actor = getattr(args, 'lr_actor', 1e-4)
    args.lr_critic = getattr(args, 'lr_critic', 1e-3)
    args.epsilon = getattr(args, 'epsilon', 0.1)
    args.noise_rate = getattr(args, 'noise_rate', 0.1)
    args.gamma = getattr(args, 'gamma', 0.95)  # 可与 common 的 gamma 区分，也可合并
    args.tau = getattr(args, 'tau', 0.01)
    args.buffer_size = getattr(args, 'buffer_size', int(5e5))
    args.batch_size = getattr(args, 'batch_size', 256)

    args.save_dir = getattr(
        args,
        'save_dir',
        args.model_dir if getattr(args, 'model_dir', '') else './model/simple_tag_6_stage1_test_2023',
    )
    args.load_dir = getattr(args, 'load_dir', './model/simple_tag_6_preTrain')
    args.save_rate = getattr(args, 'save_rate', 2000)
    args.model_dir = getattr(args, 'model_dir', '')

    args.evaluate_episodes = getattr(args, 'evaluate_episodes', args.evaluate_epoch)
    args.evaluate_episode_len = getattr(args, 'evaluate_episode_len', 100)
    args.evaluate = getattr(args, 'evaluate', False)
    args.evaluate_rate = getattr(args, 'evaluate_rate', 1000)
    args.sample_rate = getattr(args, 'sample_rate', 2000)
    args.sample_start = getattr(args, 'sample_start', 1000)

    args.high_action = args.n_actions
    # epsilon greedy from mixer
    args.epsilon = 0.5 # 1
    args.min_epsilon = 0.001 # 0.1
    anneal_steps = 50000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'

    return args

def lr_adjust(args, alg):
    if alg in ['vdn', 'qmix', 'qtran_base', 'qtran_alt', 'maven']:
        # network
        args.rnn_hidden_dim = 64
        args.qmix_hidden_dim = 32
        args.two_hyper_layers = False
        args.hyper_hidden_dim = 64
        args.qtran_hidden_dim = 64

        # epsilon greedy
        args.epsilon = 1
        args.min_epsilon = 0.5
        anneal_steps = 50000
        args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
        args.epsilon_anneal_scale = 'step'

        args.save_cycle = args.n_steps // 2

        args.lr = 6e-6

