import numpy as np

from safety.hocbf import build_hocbf_constraints, margins


class JointCBFQP:
    def __init__(self, alpha1=1.0, alpha2=1.0, teacher_margin=0.0):
        self.alpha1 = float(alpha1)
        self.alpha2 = float(alpha2)
        self.teacher_margin = float(teacher_margin)
        self.call_count = 0

    def build_constraints_from_env(self, env, active_mask=None):
        agents = env.agents
        positions = np.stack([agent.pos for agent in agents]).astype(np.float32)
        velocities = np.stack([agent.vel for agent in agents]).astype(np.float32)
        safe_radii = np.asarray([agent.safe_radius for agent in agents], dtype=np.float32)
        if active_mask is None:
            active_mask = env.get_active_agent_mask()
        collision_exempt_mask = [
            bool(env._agent_collision_exempt(agent))
            if hasattr(env, "_agent_collision_exempt")
            else False
            for agent in agents
        ]
        obstacles = [
            {"pos": np.asarray([obs.pos[0], obs.pos[1], 0.0], dtype=np.float32)[: env.dim_actions], "radius": obs.radius}
            for obs in env.obstacles
        ]
        boundaries = np.asarray([env.length, env.width, env.height], dtype=np.float32)[: env.dim_actions]
        return build_hocbf_constraints(
            positions,
            velocities,
            obstacles,
            boundaries,
            safe_radii,
            active_mask,
            collision_exempt_mask=collision_exempt_mask,
            alpha1=self.alpha1,
            alpha2=self.alpha2,
        )

    def solve_from_env(self, env, raw_joint_action, active_mask=None):
        A, c = self.build_constraints_from_env(env, active_mask=active_mask)
        raw = np.asarray(raw_joint_action, dtype=np.float32).reshape(-1)
        action_dim = int(getattr(env, "dim_actions", raw.size // len(env.agents)))
        a_max = float(getattr(env, "a_max", env.agents[0].a_max))
        active_mask = (
            np.asarray(active_mask, dtype=np.float32).reshape(-1)
            if active_mask is not None
            else np.asarray(env.get_active_agent_mask(), dtype=np.float32).reshape(-1)
        )
        corrected = self.solve(A, c, raw, a_max=a_max, active_mask=active_mask, action_dim=action_dim)
        return corrected.reshape(len(env.agents), action_dim), A, c

    def solve(self, A, c, raw_action, a_max, active_mask, action_dim):
        self.call_count += 1
        raw = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        lower = np.full_like(raw, -float(a_max))
        upper = np.full_like(raw, float(a_max))
        for idx, active in enumerate(np.asarray(active_mask).reshape(-1)):
            if active <= 0.5:
                start = idx * int(action_dim)
                lower[start : start + action_dim] = 0.0
                upper[start : start + action_dim] = 0.0
        target_margin = self.teacher_margin
        if A.size == 0:
            return np.clip(raw, lower, upper).astype(np.float32)
        if np.all(margins(A, c, np.clip(raw, lower, upper)) >= target_margin - 1e-6):
            return np.clip(raw, lower, upper).astype(np.float32)
        try:
            return self._solve_cvxpy(A, c, raw, lower, upper, target_margin)
        except Exception:
            return self._solve_scipy(A, c, raw, lower, upper, target_margin)

    @staticmethod
    def _solve_cvxpy(A, c, raw, lower, upper, target_margin):
        import cvxpy as cp

        x = cp.Variable(raw.shape[0])
        objective = cp.Minimize(cp.sum_squares(x - raw))
        constraints = [x >= lower, x <= upper, A @ x + c >= target_margin]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        if x.value is None:
            raise RuntimeError("CBF QP infeasible")
        return np.asarray(x.value, dtype=np.float32)

    @staticmethod
    def _solve_scipy(A, c, raw, lower, upper, target_margin):
        from scipy.optimize import minimize

        def objective(x):
            diff = x - raw
            return float(np.dot(diff, diff))

        constraints = [
            {"type": "ineq", "fun": lambda x, row=row, offset=offset: float(row @ x + offset - target_margin)}
            for row, offset in zip(A, c)
        ]
        result = minimize(
            objective,
            np.clip(raw, lower, upper),
            method="SLSQP",
            bounds=list(zip(lower, upper)),
            constraints=constraints,
            options={"maxiter": 100, "ftol": 1e-6, "disp": False},
        )
        if not result.success:
            raise RuntimeError(f"CBF QP infeasible: {result.message}")
        solved = np.asarray(result.x, dtype=np.float32)
        if np.any(margins(A, c, solved) < target_margin - 1e-5):
            raise RuntimeError("CBF QP solver returned a constraint-violating action")
        return solved
