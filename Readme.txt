运行 main.py 进行训练
在 arguments.py 中选择算法和环境
目前支持 vdn, qmix, ,mappo, macpo, rgmcomm 五种算法（ --alg)
--map Basic2P 是 迷宫环境，其他的星际环境命名遵照 smac 原始环境，代码中记录了所有的字典
新增 UAV 环境入口：
--map UAV2D 运行二维离散动作多无人机寻路环境
--map UAV3D 运行三维离散动作多无人机寻路环境（障碍物按圆柱处理）
新增 MPE2 协作环境入口：
--map simple_speaker_listener 运行 MPE2 / PettingZoo 的 simple_speaker_listener_v4 场景

运行示例：
python main.py --alg mappo --map UAV3D
python main.py --alg mappo_reshape --map UAV3D
python main.py --alg mappo --map simple_speaker_listener

算法名后 ‘_reshape’ 是使用 G 网络，‘_Comm’ 是使用 C 网络，二者可叠加
