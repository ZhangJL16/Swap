import types
import os
import tempfile

import numpy as np
import torch

from common.rollout import RolloutWorker
from common.replay_buffer import ReplayBuffer
from hierarchy.energy_buffer import EnergyReplayBuffer
from envs.UAVEnergyDeliveryHierarchical import UAVEnvDiscreteWrapper
from hierarchy.high_level_controller import order_features
from hierarchy.navigation_energy import NavigationEnergyNetwork
from hierarchy.order_replay_buffer import OrderReplayBuffer
from hierarchy.order_q import OrderQNetwork
from safety.correction_buffer import CorrectionBuffer
from safety.constrained_flow import ConstrainedFlow
from safety.distilled_corrector import DistilledCorrector
from safety.flow_trainer import FlowTrainer
from safety.hocbf import build_hocbf_constraints, margins
from safety.joint_cbf_qp import JointCBFQP


def make_env(episode_limit=8):
    return UAVEnvDiscreteWrapper(
        dim_actions=3,
        num_hunters=3,
        episode_limit=episode_limit,
        total_orders=4,
        max_active_orders=2,
        num_obstacle=0,
        cbf_flow_enabled=True,
        agent_entry_interval=1,
        order_max_duration=5,
        energy_reserve=0.0,
        initial_energy=100.0,
    )


def test_cbf_flow_reset_and_entry_schedule():
    env = make_env()
    env.reset(seed=7)
    station = env.env.charging_station_positions[0]
    for agent in env.env.agents:
        np.testing.assert_allclose(agent.pos, station)
        assert not agent.active

    energies = np.asarray([agent.energy for agent in env.env.agents], dtype=np.float32)
    env.prepare_cbf_flow_step()
    assert env.get_active_agent_mask().tolist() == [1.0, 0.0, 0.0]
    after_prepare_energy = np.asarray([agent.energy for agent in env.env.agents], dtype=np.float32)
    np.testing.assert_allclose(after_prepare_energy, energies)

    env.step([np.zeros(3, dtype=np.float32) for _ in range(env.n_agents)])
    env.prepare_cbf_flow_step()
    assert env.get_active_agent_mask().tolist() == [1.0, 1.0, 0.0]


def test_disabled_cbf_flow_keeps_legacy_active_agents():
    env = UAVEnvDiscreteWrapper(
        dim_actions=3,
        num_hunters=3,
        episode_limit=4,
        total_orders=4,
        max_active_orders=2,
        num_obstacle=0,
        cbf_flow_enabled=False,
    )
    env.reset(seed=8)
    assert env.env.cbf_flow_enabled is False
    np.testing.assert_allclose(env.get_active_agent_mask(), np.ones(env.n_agents, dtype=np.float32))


def test_dynamic_orders_do_not_terminal_on_all_completed():
    env = make_env()
    env.reset(seed=9)
    env.prepare_cbf_flow_step()
    assert not env.env._all_orders_completed()
    assert len(env.env.active_order_ids) == env.env.max_active_orders
    old_next_id = env.env.next_order_id_to_activate
    order_id = env.env.active_order_ids[0]
    env.env.orders[order_id].status = env.env.orders[order_id].COMPLETED
    env.env._remove_active_order(order_id)
    env.env._activate_orders()
    assert len(env.env.active_order_ids) == env.env.max_active_orders
    assert env.env.next_order_id_to_activate > old_next_id


def test_cbf_flow_max_active_orders_is_not_clipped_by_total_orders():
    env = UAVEnvDiscreteWrapper(
        dim_actions=3,
        num_hunters=2,
        episode_limit=4,
        total_orders=1,
        max_active_orders=3,
        num_obstacle=0,
        cbf_flow_enabled=True,
        agent_entry_interval=1,
    )
    env.reset(seed=10)
    env.prepare_cbf_flow_step()
    assert env.env.max_active_orders == 3
    assert len(env.env.active_order_ids) == 3
    assert env.env.summary()["total_orders"] == 3.0


def test_high_decision_first_come_order_assignment_and_no_wait_action():
    env = make_env()
    env.reset(seed=11)
    env.prepare_cbf_flow_step()
    assert env.env.high_level_n_actions == 0
    assert env.env.high_level_mode_n_actions == 0
    orders_for_agent0 = env.env.get_visible_available_orders(0)
    assert orders_for_agent0
    order_id = orders_for_agent0[0].order_id
    assert env.env.assign_order(0, order_id)
    assert order_id not in [o.order_id for o in env.env.get_visible_available_orders(1)]


def test_order_q_and_navigation_energy_strict_input_dims_and_variable_orders():
    order_q = OrderQNetwork(hidden_dim=16)
    assert order_q.input_dim == 13
    uav_state = torch.zeros(4, 6)
    order_state = torch.zeros(4, 7)
    assert order_q(uav_state, order_state).shape == (4,)
    assert order_q(torch.zeros(1, 6), torch.zeros(1, 7)).shape == (1,)

    energy = NavigationEnergyNetwork(hidden_dim=16)
    assert energy.input_dim == 9
    out = energy(torch.zeros(3, 3), torch.zeros(3, 3), torch.zeros(3, 3))
    assert out.shape == (3, 1)
    assert torch.all(out >= 0)


def test_order_features_are_absolute_13d_components():
    env = make_env()
    env.reset(seed=13)
    env.prepare_cbf_flow_step()
    order = env.env.get_visible_available_orders(0)[0]
    features = order_features(order, default_time_limit=5)
    assert features.shape == (7,)
    q_input = np.concatenate([env.env.high_controller.uav_state(env.env.agents[0]), features])
    assert q_input.shape == (13,)


def test_cbf_qp_satisfies_hocbf_constraints():
    positions = np.asarray([[0.02, 2.0, 2.0]], dtype=np.float32)
    velocities = np.asarray([[-0.1, 0.0, 0.0]], dtype=np.float32)
    obstacles = []
    boundaries = np.asarray([4.0, 4.0, 4.0], dtype=np.float32)
    safe_radii = np.asarray([0.01], dtype=np.float32)
    active = np.asarray([1.0], dtype=np.float32)
    A, c = build_hocbf_constraints(
        positions, velocities, obstacles, boundaries, safe_radii, active, alpha1=2.0, alpha2=2.0
    )
    raw = np.asarray([-1.0, 0.0, 0.0], dtype=np.float32)
    qp = JointCBFQP(alpha1=2.0, alpha2=2.0, teacher_margin=0.0)
    solved = qp.solve(A, c, raw, a_max=1.0, active_mask=active, action_dim=3)
    assert np.all(margins(A, c, solved) >= -1e-5)


def test_flow_cbf_loss_has_nonzero_finite_gradients():
    flow = ConstrainedFlow(safety_state_dim=5, joint_action_dim=3, hidden_dim=16)
    for param in flow.parameters():
        torch.nn.init.zeros_(param)
    trainer = FlowTrainer(flow, flow_train_steps=2, flow_margin=0.5, kappa=1.0)
    batch = {
        "safety_state_abs": np.zeros((2, 5), dtype=np.float32),
        "raw_joint_action": np.zeros((2, 3), dtype=np.float32),
        "correct_joint_action": np.ones((2, 3), dtype=np.float32) * 0.5,
        "constraint_A": np.asarray([[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]], dtype=np.float32),
        "constraint_c": np.zeros((2, 1), dtype=np.float32),
    }
    losses = trainer.losses(batch)
    trainer.optimizer.zero_grad()
    losses["cbf_loss"].backward()
    grads = [
        p.grad.detach()
        for p in flow.parameters()
        if p.grad is not None and p.grad.detach().abs().sum().item() > 0.0
    ]
    assert grads
    assert all(torch.isfinite(g).all().item() for g in grads)


def test_flow_integral_endpoint_can_match_teacher_after_training():
    torch.manual_seed(0)
    flow = ConstrainedFlow(safety_state_dim=4, joint_action_dim=2, hidden_dim=32)
    trainer = FlowTrainer(flow, lr=1e-2, flow_train_steps=4, cbf_coef=0.0, terminal_coef=0.0)
    batch = {
        "safety_state_abs": np.zeros((8, 4), dtype=np.float32),
        "raw_joint_action": np.zeros((8, 2), dtype=np.float32),
        "correct_joint_action": np.ones((8, 2), dtype=np.float32) * 0.25,
        "constraint_A": np.zeros((8, 1, 2), dtype=np.float32),
        "constraint_c": np.ones((8, 1), dtype=np.float32),
    }
    for _ in range(80):
        trainer.train_step(batch)
    with torch.no_grad():
        end = flow.integrate(torch.zeros(8, 4), torch.zeros(8, 2), steps=4)
    assert torch.mean((end - 0.25).pow(2)).item() < 2e-3


def test_student_is_single_forward_and_does_not_call_qp():
    student = DistilledCorrector(safety_state_dim=5, joint_action_dim=3, hidden_dim=16)
    qp = JointCBFQP()
    before = qp.call_count
    action = student(torch.zeros(1, 5), torch.zeros(1, 3))
    assert action.shape == (1, 3)
    assert qp.call_count == before


class DummyAgents:
    def __init__(self, env):
        self.env = env
        self.n_agents = env.n_agents
        self.last_guard_applied = [0.0 for _ in range(env.n_agents)]
        self.cbf_teacher = JointCBFQP(alpha1=1.0, alpha2=1.0, teacher_margin=0.0)
        self.policy = types.SimpleNamespace(init_hidden=lambda episode_num: None)

    def reset_episode_state(self):
        pass

    def choose_action(self, obs, last_action, agent_id, avail_action, epsilon, **kwargs):
        del obs, last_action, agent_id, avail_action, epsilon, kwargs
        return np.zeros(3, dtype=np.float32)

    def get_low_action_log_probs(self, observations, actions):
        del observations, actions
        return np.zeros((self.n_agents, 1), dtype=np.float32)

    def revise_safe_actions(self, observations, avail_actions, base_actions):
        del observations, avail_actions
        active = self.env.get_active_agent_mask()
        corrected, _, _ = self.cbf_teacher.solve_from_env(
            self.env.env, np.asarray(base_actions, dtype=np.float32), active_mask=active
        )
        raw = np.asarray(base_actions, dtype=np.float32)
        self.last_guard_applied = (np.linalg.norm(corrected - raw, axis=-1) > 1e-6).astype(np.float32).tolist()
        return corrected.astype(np.float32)


def test_rollout_saves_raw_correct_and_keeps_intervened_samples_in_source():
    env = make_env(episode_limit=3)
    info = env.get_env_info()
    args = types.SimpleNamespace(
        alg="hmappo_cbf_flow",
        map="UAVEnergyDeliveryHierarchical",
        episode_limit=3,
        n_actions=info["n_actions"],
        n_agents=info["n_agents"],
        state_shape=info["state_shape"],
        obs_shape=info["obs_shape"],
        low_action_type="continuous",
        low_action_dim=3,
        epsilon=0.0,
        anneal_epsilon=0.0,
        min_epsilon=0.0,
        n_steps=3,
        replay_dir="",
        evaluate_epoch=1,
        epsilon_anneal_scale="episode",
        gamma=0.99,
        is_level_training=False,
        high_level_n_actions=0,
        high_level_obs_shape=0,
        high_level_state_shape=0,
        seed=1,
        eval_seed=2,
        cuda=False,
    )
    worker = RolloutWorker(env, DummyAgents(env), args)
    episode, *_ = worker.generate_episode(evaluate=False)
    assert "u_raw" in episode and "u_correct" in episode
    assert episode["u_raw"].shape[-1] == 3
    assert episode["u_correct"].shape[-1] == 3
    assert "agent_actor_mask = agent_mask" in open("level_policy/mappo.py", encoding="utf-8").read()


def test_replay_buffer_persists_raw_correct_flow_fields():
    args = types.SimpleNamespace(
        alg="hmappo_cbf_flow",
        n_actions=3,
        low_action_type="continuous",
        low_action_dim=3,
        n_agents=2,
        state_shape=5,
        obs_shape=4,
        buffer_size=2,
        episode_limit=3,
    )
    buffer = ReplayBuffer(args)
    episode = {
        "o": np.zeros((1, 3, 2, 4), dtype=np.float32),
        "u": np.zeros((1, 3, 2, 3), dtype=np.float32),
        "s": np.zeros((1, 3, 5), dtype=np.float32),
        "r": np.zeros((1, 3, 1), dtype=np.float32),
        "o_next": np.zeros((1, 3, 2, 4), dtype=np.float32),
        "s_next": np.zeros((1, 3, 5), dtype=np.float32),
        "avail_u": np.zeros((1, 3, 2, 3), dtype=np.float32),
        "avail_u_next": np.zeros((1, 3, 2, 3), dtype=np.float32),
        "u_onehot": np.zeros((1, 3, 2, 3), dtype=np.float32),
        "padded": np.zeros((1, 3, 1), dtype=np.float32),
        "terminated": np.zeros((1, 3, 1), dtype=np.float32),
        "warning_signal": np.zeros((1, 3, 2, 1), dtype=np.float32),
        "agent_active_mask": np.ones((1, 3, 2, 1), dtype=np.float32),
        "u_raw": np.ones((1, 3, 2, 3), dtype=np.float32),
        "u_correct": np.ones((1, 3, 2, 3), dtype=np.float32) * 2.0,
        "raw_log_prob": np.ones((1, 3, 2, 1), dtype=np.float32) * -0.5,
        "correction_delta": np.ones((1, 3, 2, 3), dtype=np.float32),
        "intervention_mask": np.ones((1, 3, 2, 1), dtype=np.float32),
    }
    buffer.store_episode(episode)
    sample = buffer.sample(1)
    np.testing.assert_allclose(sample["u_raw"], 1.0)
    np.testing.assert_allclose(sample["u_correct"], 2.0)
    np.testing.assert_allclose(sample["raw_log_prob"], -0.5)


def test_stage_buffers_roundtrip():
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_path = tmp_dir.name
    correction = CorrectionBuffer(8)
    correction.add(
        {
            "safety_state_abs": np.zeros(5, dtype=np.float32),
            "raw_joint_action": np.zeros(3, dtype=np.float32),
            "correct_joint_action": np.ones(3, dtype=np.float32),
            "constraint_A": np.zeros((1, 3), dtype=np.float32),
            "constraint_c": np.ones(1, dtype=np.float32),
            "active_mask": np.ones(1, dtype=np.float32),
        }
    )
    path = os.path.join(tmp_path, "correction.pkl")
    correction.save(path)
    loaded_correction = CorrectionBuffer(1)
    loaded_correction.load(path)
    assert len(loaded_correction) == 1
    np.testing.assert_allclose(loaded_correction.storage[0]["correct_joint_action"], 1.0)

    energy = EnergyReplayBuffer(8)
    energy.add(
        {
            "position": np.zeros(3, dtype=np.float32),
            "velocity": np.zeros(3, dtype=np.float32),
            "target": np.ones(3, dtype=np.float32),
            "step_energy": 0.5,
            "next_position": np.ones(3, dtype=np.float32),
            "next_velocity": np.zeros(3, dtype=np.float32),
            "goal_done": 0.0,
            "loaded_leg": 1.0,
        }
    )
    path = os.path.join(tmp_path, "energy.pkl")
    energy.save(path)
    loaded_energy = EnergyReplayBuffer(1)
    loaded_energy.load(path)
    assert len(loaded_energy) == 1
    assert loaded_energy.storage[0]["step_energy"] == 0.5

    order_buffer = OrderReplayBuffer(8)
    order_buffer.add(
        {
            "uav_state": np.zeros(6, dtype=np.float32),
            "order": np.ones(7, dtype=np.float32),
            "option_return": 1.0,
            "duration": 2.0,
            "next_uav_state": np.ones(6, dtype=np.float32),
            "next_feasible_orders": [np.ones(7, dtype=np.float32)],
            "episode_done": 0.0,
        }
    )
    path = os.path.join(tmp_path, "order.pkl")
    order_buffer.save(path)
    loaded_order = OrderReplayBuffer(1)
    loaded_order.load(path)
    assert len(loaded_order) == 1
    np.testing.assert_allclose(loaded_order.storage[0]["order"], 1.0)
    tmp_dir.cleanup()
