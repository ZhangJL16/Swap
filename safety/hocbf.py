import numpy as np


def _as_dim_vector(value, dim):
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < dim:
        arr = np.pad(arr, (0, dim - arr.size))
    return arr[:dim].astype(np.float32)


def add_hocbf_constraint(rows, offsets, coeffs, h, h_dot, alpha1, alpha2):
    rows.append(coeffs.astype(np.float32))
    offsets.append(float(2.0 * 0.0 + (alpha1 + alpha2) * h_dot + alpha1 * alpha2 * h))


def build_hocbf_constraints(
    positions,
    velocities,
    obstacles,
    boundaries,
    safe_radii,
    active_mask,
    collision_exempt_mask=None,
    alpha1=1.0,
    alpha2=1.0,
):
    """Build A, c for HOCBF constraints A a + c >= 0.

    Relative quantities are computed only inside this function.
    """

    positions = np.asarray(positions, dtype=np.float32)
    velocities = np.asarray(velocities, dtype=np.float32)
    safe_radii = np.asarray(safe_radii, dtype=np.float32).reshape(-1)
    active_mask = np.asarray(active_mask, dtype=np.float32).reshape(-1) > 0.5
    if collision_exempt_mask is None:
        collision_exempt_mask = np.zeros_like(active_mask, dtype=bool)
    else:
        collision_exempt_mask = (
            np.asarray(collision_exempt_mask, dtype=np.float32).reshape(-1) > 0.5
        )
    n_agents, dim = positions.shape
    action_dim = n_agents * dim
    rows = []
    offsets = []
    boundaries = _as_dim_vector(boundaries, dim)

    for i in range(n_agents):
        if not active_mask[i]:
            continue
        p_i = positions[i]
        v_i = velocities[i]
        r_i = float(safe_radii[i])

        for obstacle in obstacles:
            obs_pos = _as_dim_vector(obstacle["pos"], dim)
            obs_radius = float(obstacle["radius"])
            diff = p_i - obs_pos
            radius = r_i + obs_radius
            h = float(np.dot(diff, diff) - radius ** 2)
            h_dot = float(2.0 * np.dot(diff, v_i))
            coeff = np.zeros(action_dim, dtype=np.float32)
            coeff[i * dim : (i + 1) * dim] = 2.0 * diff
            c = float(2.0 * np.dot(v_i, v_i) + (alpha1 + alpha2) * h_dot + alpha1 * alpha2 * h)
            rows.append(coeff)
            offsets.append(c)

        for axis, boundary in enumerate(boundaries):
            coeff = np.zeros(action_dim, dtype=np.float32)
            h_low = float(p_i[axis] - r_i)
            h_dot_low = float(v_i[axis])
            coeff[i * dim + axis] = 1.0
            rows.append(coeff.copy())
            offsets.append(float((alpha1 + alpha2) * h_dot_low + alpha1 * alpha2 * h_low))

            h_high = float(boundary - r_i - p_i[axis])
            h_dot_high = float(-v_i[axis])
            coeff[i * dim + axis] = -1.0
            rows.append(coeff.copy())
            offsets.append(float((alpha1 + alpha2) * h_dot_high + alpha1 * alpha2 * h_high))

    for i in range(n_agents):
        if not active_mask[i]:
            continue
        for j in range(i + 1, n_agents):
            if not active_mask[j]:
                continue
            if collision_exempt_mask[i] and collision_exempt_mask[j]:
                continue
            diff = positions[i] - positions[j]
            rel_v = velocities[i] - velocities[j]
            radius = float(safe_radii[i] + safe_radii[j])
            h = float(np.dot(diff, diff) - radius ** 2)
            h_dot = float(2.0 * np.dot(diff, rel_v))
            coeff = np.zeros(action_dim, dtype=np.float32)
            coeff[i * dim : (i + 1) * dim] = 2.0 * diff
            coeff[j * dim : (j + 1) * dim] = -2.0 * diff
            c = float(
                2.0 * np.dot(rel_v, rel_v)
                + (alpha1 + alpha2) * h_dot
                + alpha1 * alpha2 * h
            )
            rows.append(coeff)
            offsets.append(c)

    if not rows:
        return np.zeros((0, action_dim), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.stack(rows).astype(np.float32), np.asarray(offsets, dtype=np.float32)

def margins(A, c, action):
    return np.asarray(A, dtype=np.float32) @ np.asarray(action, dtype=np.float32).reshape(-1) + np.asarray(c, dtype=np.float32)
