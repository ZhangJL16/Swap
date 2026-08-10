import copy
import multiprocessing as mp
import os
from contextlib import contextmanager

from agent.agent import Agents
from common.rollout import RolloutWorker, smac_penalty_enabled


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


def _build_worker_smac_env(args):
    from smac.env import StarCraft2Env

    with _suppress_stdio(not bool(args.debug)):
        base_env = StarCraft2Env(
            map_name=args.map,
            step_mul=args.step_mul,
            difficulty=args.difficulty,
            game_version=args.game_version,
            replay_dir="",
            debug=args.debug,
        )
    base_env = _SilentSMACEnv(base_env, enabled=not bool(args.debug))

    if smac_penalty_enabled(args):
        from envs.SMACSafeEnv import SMACSafetyWrapper

        return SMACSafetyWrapper(
            base_env,
            risk_threshold=float(getattr(args, "guard_risk_threshold", 0.1)),
        )
    return base_env


def _smac_worker_main(conn, args, worker_id):
    worker_args = copy.deepcopy(args)
    worker_args.cuda = False
    worker_args.load_model = False
    worker_args.replay_dir = ""

    env = _build_worker_smac_env(worker_args)
    agents = Agents(worker_args, env)
    rollout = RolloutWorker(env, agents, worker_args)

    try:
        while True:
            message = conn.recv()
            cmd = message.get("cmd")
            if cmd == "close":
                break
            if cmd != "collect":
                raise ValueError(f"Unknown worker command: {cmd}")

            agents.load_rollout_state(message.get("snapshot", {}))
            rollout.epsilon = float(message.get("epsilon", rollout.epsilon))
            episode, episode_reward, summary, steps, _ = rollout.generate_episode(
                episode_num=worker_id,
                evaluate=False,
            )
            conn.send(
                {
                    "episode": episode,
                    "episode_reward": episode_reward,
                    "summary": summary,
                    "steps": steps,
                    "epsilon_after": float(rollout.epsilon),
                }
            )
    finally:
        try:
            env.close()
        finally:
            conn.close()


class ParallelSMACEpisodeCollector:
    def __init__(self, args, agents):
        self.args = args
        self.agents = agents
        self.n_workers = max(1, int(getattr(args, "smac_parallel_envs", 1)))
        self.ctx = mp.get_context("spawn")
        self.parents = []
        self.processes = []
        for worker_id in range(self.n_workers):
            parent_conn, child_conn = self.ctx.Pipe()
            process = self.ctx.Process(
                target=_smac_worker_main,
                args=(child_conn, args, worker_id),
                daemon=True,
            )
            process.start()
            child_conn.close()
            self.parents.append(parent_conn)
            self.processes.append(process)
        self.epsilon = float(getattr(args, "epsilon", 0.0))
        self.anneal_epsilon = float(getattr(args, "anneal_epsilon", 0.0))
        self.min_epsilon = float(getattr(args, "min_epsilon", 0.0))
        self.epsilon_anneal_scale = getattr(args, "epsilon_anneal_scale", "step")

    def close(self):
        for parent in self.parents:
            try:
                parent.send({"cmd": "close"})
            except Exception:
                pass
        for parent in self.parents:
            try:
                parent.close()
            except Exception:
                pass
        for process in self.processes:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _advance_epsilon(self, steps, episode_count):
        if self.epsilon_anneal_scale == "step":
            self.epsilon = max(
                self.min_epsilon,
                self.epsilon - self.anneal_epsilon * int(steps),
            )
        elif self.epsilon_anneal_scale == "episode":
            self.epsilon = max(
                self.min_epsilon,
                self.epsilon - self.anneal_epsilon * int(episode_count),
            )

    def collect_episodes(self, n_episodes):
        snapshot = self.agents.export_rollout_state()
        remaining = int(n_episodes)
        results = []
        while remaining > 0:
            active = min(self.n_workers, remaining)
            epsilon_for_wave = self.epsilon
            for worker_idx in range(active):
                self.parents[worker_idx].send(
                    {
                        "cmd": "collect",
                        "snapshot": snapshot,
                        "epsilon": epsilon_for_wave,
                    }
                )
            for worker_idx in range(active):
                result = self.parents[worker_idx].recv()
                results.append(result)
                self._advance_epsilon(result.get("steps", 0), 1)
            remaining -= active
        return results
