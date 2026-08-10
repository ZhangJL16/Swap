# UAV2D 使用说明

这个项目包含多种多智能体强化学习环境和算法入口；如果只使用 `UAV2D`，可以忽略 StarCraft II/SMAC、MPE2、UAV3D 和 UAVEncircle。`UAV2D` 的实际实现不在单独文件里，而是在 `envs/UAVEnv.py` 中通过 `dim_actions=2` 开启二维模式。

## 代码路径

- `envs/UAVEnv.py`：UAV 环境主体。
  - `UAVEnv`：连续动作动力学环境。
  - `UAVEnvDiscreteWrapper`：把连续二维加速度动作离散化，供本项目的 MARL 算法训练使用。
  - `parallel_env` / `UAVParallelEnv`：字典式多智能体接口，适合自己写环境交互脚本。
- `main.py`：训练入口。`--map UAV2D` 会构造 `UAVEnvDiscreteWrapper(dim_actions=2, num_hunters=args.uav_n_agents)`。
- `common/arguments.py`：命令行参数，UAV 数量由 `--uav_n_agents` 控制，默认 4。
- `runner.py` 和 `common/rollout.py`：训练、评估、日志和模型保存流程。
- `test_model.py`：加载训练好的模型并生成可视化 GIF/逐帧图片。
- `plot_train_logs.py`：从 `train_logs` 下的 CSV 绘制训练曲线。

## 环境安装

建议在项目根目录运行。当前项目没有 `setup.py` 或 `pyproject.toml`，脚本需要从仓库根目录启动。

```bash
cd /home/zjl/MARL
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install numpy gymnasium matplotlib torch scipy pandas tqdm pillow networkx
```

如果你需要完整复现项目依赖，可以尝试：

```bash
python3 -m pip install -r requirements.uv.txt
```

`requirements.txt` 是 UTF-16 文本；更建议使用 `requirements.uv.txt`。只跑 `UAV2D` 不需要安装 StarCraft II，也不需要配置 `SC2PATH`。

## 快速训练 UAV2D

推荐先用 `mappo` 跑通流程：

```bash
python3 main.py \
  --alg mappo \
  --map UAV2D \
  --uav_n_agents 4 \
  --n_steps 200000 \
  --evaluate_cycle 5000 \
  --evaluate_epoch 5
```

训练输出：

- 模型：`model/mappo/UAV2D/`
- CSV 日志：`train_logs/mappo/UAV2D/UAV2D_log_<n>.csv`
- `.mat` 结果：`result/UAV2D/mappo/`

也可以换成项目已有的其他离散动作算法，例如：

```bash
python3 main.py --alg qmix --map UAV2D --uav_n_agents 4
python3 main.py --alg vdn --map UAV2D --uav_n_agents 4
python3 main.py --alg mappo_safe --map UAV2D --uav_n_agents 4
```

## 单独运行某个 UAVDelivery 方法

如果只想上传一个脚本文件并逐个方法运行，上传 `scripts/UAVDelivery/run_method.sh` 即可。脚本默认使用之前 `mappo` 的 UAVDelivery 参数：

```bash
--map UAVDelivery
--uav_n_agents 4
--uav_total_orders 8
--uav_max_active_orders 4
--seed 123
--eval_seed 100123
--evaluate_epoch 20
--cuda True
--gpu_id 0
--experiment_device ${EXPERIMENT_DEVICE:-${MARL_EXPERIMENT_DEVICE:-dorm}}
```

运行某个方法时只需要把方法名写在脚本后面：

```bash
./scripts/UAVDelivery/run_method.sh mappo
./scripts/UAVDelivery/run_method.sh qmix
./scripts/UAVDelivery/run_method.sh vdn
./scripts/UAVDelivery/run_method.sh macpo
./scripts/UAVDelivery/run_method.sh rgmcomm
```

也可以显式使用 `--alg`：

```bash
./scripts/UAVDelivery/run_method.sh --alg mappo
```

需要临时覆盖参数时，把参数继续接在后面即可；没有覆盖的参数仍沿用上面的默认值：

```bash
./scripts/UAVDelivery/run_method.sh qmix --gpu_id 1 --n_steps 600000 --evaluate_cycle 20
./scripts/UAVDelivery/run_method.sh mappo_safe_Comm --n_steps 600000
```

脚本默认使用 `.venv/bin/python3`，日志写到 `logs/uav_delivery_single_method/<timestamp>/<alg>.log`。

为了公平对比，脚本默认固定训练随机种子 `--seed 123`，并固定评估场景种子 `--eval_seed 100123`。评估默认用 `--evaluate_epoch 20` 个固定场景取平均；所有算法会用同一组评估场景。需要换一组可复现实验时，可以用环境变量覆盖：

```bash
SEED=456 EVALUATE_EPOCH=20 ./scripts/UAVDelivery/run_method.sh mappo
SEED=456 EVALUATE_EPOCH=20 ./scripts/UAVDelivery/run_all_methods.sh
```

注意：`common/arguments.py` 里部分布尔参数使用 `type=bool`，命令行传 `--cuda False` 在 Python 中仍可能被解析成 `True`。如果只用 `mappo`，代码会在 CUDA 不可用时自动落到 CPU；如果其他算法在 CPU 机器上因为 CUDA 报错，需要把 `common/arguments.py` 中 `--cuda` 的默认值改成 `False`，或改成标准的 `store_true/store_false` 写法。

## 评估和可视化

训练完成后，可以加载 `model/<alg>/UAV2D/` 下的模型：

```bash
python3 test_model.py \
  --alg mappo \
  --map UAV2D \
  --uav-n-agents 4 \
  --episodes 3 \
  --render \
  --render-mode rgb_array
```

输出位置：

- GIF：`result/test/mappo/UAV2D/`
- 每帧图片：`test_result/mappo/UAV2D/episode_<n>/`

绘制训练日志：

```bash
python3 plot_train_logs.py \
  --map UAV2D \
  --algs mappo \
  --metrics episode_reward,win_rate,episode_steps \
  --latest-log \
  --smooth 10
```

图片默认保存到 `plots/`。

## UAV2D 环境逻辑

默认环境参数来自 `envs/UAVEnv.py`：

- 地图大小：`4.0 x 4.0`
- UAV 数量：训练入口默认 4，可用 `--uav_n_agents` 修改
- 障碍物数量：10 个圆形障碍物
- 每个 UAV 有一个独立目标点
- 每回合上限：200 步
- 目标到达阈值：`0.12`
- UAV 安全半径：`0.05`
- 最大速度：`0.16`
- 最大加速度：`0.05`
- 激光距离传感器：16 条射线，最大距离 `0.35`

每次 `reset(seed=...)` 会随机生成障碍物、UAV 初始位置和目标点，并保证它们之间留出最小安全距离。二维环境中的障碍物按圆处理；三维模式才把同一套障碍物渲染成圆柱。

## 动作空间

项目训练默认使用 `UAVEnvDiscreteWrapper`。二维离散动作数为 `3^2 = 9`，动作索引来自：

```python
itertools.product([-1.0, 0.0, 1.0], repeat=2)
```

非零方向会先归一化，再乘以 `a_max=0.05`，作为二维加速度输入到底层连续环境。动作顺序为：

| 索引 | 方向 |
| --- | --- |
| 0 | `(-1, -1)` |
| 1 | `(-1, 0)` |
| 2 | `(-1, 1)` |
| 3 | `(0, -1)` |
| 4 | `(0, 0)` |
| 5 | `(0, 1)` |
| 6 | `(1, -1)` |
| 7 | `(1, 0)` |
| 8 | `(1, 1)` |

底层 `UAVEnv` 也支持连续动作：每个智能体传入 shape 为 `(2,)` 的加速度向量，环境会按最大加速度和最大速度裁剪。

## 观测、状态和消息

对 `UAV2D`：

- 单智能体观测维度：23
- 离散动作数：9
- `msg_shape`：6
- 默认 4 个 UAV、10 个障碍物时，全局状态维度：58

单个智能体观测结构：

```text
[pos_xy_norm(2), vel_xy_norm(2), lasers(16), goal_delta_xy_norm(2), goal_distance_norm(1)]
```

全局状态结构：

```text
每个 UAV: [pos_xy_norm(2), vel_xy_norm(2), goal_xy_norm(2), reached(1)]
每个障碍物: [obstacle_xy_norm(2), obstacle_radius_norm(1)]
```

消息 `get_msg()` 用于通信类算法。对每个接收者，返回其他智能体的：

```text
[sender_goal_delta_xy_norm(2), sender_velocity_xy_norm(2), sender_risk(1), sender_reached(1)]
```

因此形状是 `(n_agents - 1, 6)`。

## 奖励和终止

底层环境每步为每个 UAV 计算奖励，离散包装器返回所有 UAV 的平均奖励。

奖励主要由这些部分组成：

- 接近目标的距离进度：`+2.5 * progress`
- 朝向目标的速度奖励：`+0.9 * positive_velocity_toward_goal`
- 每步时间成本：`-0.01`
- 边界/障碍物碰撞惩罚：`-1.2`
- UAV 之间碰撞惩罚：`-1.5`
- 风险惩罚：`-0.2 * reward_safe_value`
- 距离目标惩罚：`-0.3 * min(distance_to_goal, 1.0)`
- 单个 UAV 到达目标：`+8.0`
- 全部 UAV 到达目标：所有 UAV 额外 `+5.0`

`terminated=True` 的条件：

- 所有 UAV 都到达各自目标；或
- 回合步数达到 `episode_limit`。

`info["battle_won"]` 表示是否所有 UAV 成功到达目标。`info["warning_signal"]` 是每个 UAV 的安全风险值，来自边界、障碍物和 UAV 间距离的非线性重叠计算。`info["per_agent_reward"]` 保存每个 UAV 的原始奖励。

## 直接调用离散环境

如果只想把 UAV2D 当作环境使用：

```python
import numpy as np
from envs.UAVEnv import UAVEnvDiscreteWrapper

env = UAVEnvDiscreteWrapper(dim_actions=2, num_hunters=4, episode_limit=200)

obs = env.reset(seed=42)
print(env.get_env_info())
print(obs.shape)          # (4, 23)
print(env.get_state().shape)

terminated = False
while not terminated:
    actions = np.random.randint(env.n_actions, size=env.n_agents)
    reward, terminated, info = env.step(actions)
    obs = env.get_obs()

print(reward, info["battle_won"], env.summary())
env.close()
```

注意：`UAVEnvDiscreteWrapper.step()` 返回的是三元组：

```python
reward, terminated, info = env.step(actions)
```

它不是 Gymnasium 标准的五元组；下一步观测需要再调用 `env.get_obs()`。

## 直接调用连续环境

如果你的算法输出连续二维加速度：

```python
import numpy as np
from envs.UAVEnv import UAVEnv

env = UAVEnv(dim_actions=2, num_hunters=4, episode_limit=200)
obs = env.reset(seed=42)

actions = [np.zeros(2, dtype=np.float32) for _ in env.agents]
obs, per_agent_rewards, reached_flags, safe_value = env.step(actions)

print(obs.shape, per_agent_rewards, reached_flags, safe_value)
env.close()
```

## 字典式多智能体接口

`parallel_env` 更接近 PettingZoo ParallelEnv 的使用方式：

```python
from envs.UAVEnv import parallel_env

env = parallel_env(
    dim_actions=2,
    num_hunters=4,
    continuous_actions=False,
    render_mode=None,
)

obs, infos = env.reset(seed=42)
actions = {
    agent: env.action_space(agent).sample()
    for agent in env.agents
}
obs, rewards, terminated, truncated, infos = env.step(actions)

env.close()
```

如果设置 `continuous_actions=True`，每个 agent 的动作空间是 `Box(-0.05, 0.05, shape=(2,))`。

## 渲染

离散包装器本身没有 `render()`，需要通过底层环境渲染：

```python
from PIL import Image
from envs.UAVEnv import UAVEnvDiscreteWrapper

env = UAVEnvDiscreteWrapper(dim_actions=2, num_hunters=4)
env.reset(seed=42)

frame = env.env.render(show=False)
Image.fromarray(frame).save("uav2d_debug.png")

env.close()
```

`show=True` 会打开 matplotlib 交互窗口；服务器或无显示环境中建议使用 `show=False`。

## 已知注意点

- `UAVEnvDiscreteWrapper()` 类默认 `num_hunters=8`，但 `main.py --map UAV2D` 默认使用 `--uav_n_agents=4`。直接写脚本时建议显式传 `num_hunters`。
- `runner.py` 中 `UAV_COLLISION_MAPS` 当前只包含 `UAV3D` 和 `UAVEncircle`，所以 UAV2D 训练 CSV 里的碰撞列默认可能为空；环境自身 `summary()` 仍会统计 `collision_count`、`obstacle_collision_count`、`agent_collision_count`。
- 所有脚本都假设从项目根目录运行，否则相对路径下的 `model/`、`result/`、`train_logs/` 可能不符合预期。
