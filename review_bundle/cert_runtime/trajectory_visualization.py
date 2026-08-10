from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

from matplotlib import animation
from matplotlib import pyplot as plt
import numpy as np

from envs.certified_uav.scenario import ScenarioDefinition


def read_trajectory(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError("trajectory is empty")
    return records


def _draw_box(ax, low: np.ndarray, high: np.ndarray, color: str = "0.35") -> None:
    corners = np.array([
        [low[0], low[1], low[2]], [high[0], low[1], low[2]],
        [high[0], high[1], low[2]], [low[0], high[1], low[2]],
        [low[0], low[1], high[2]], [high[0], low[1], high[2]],
        [high[0], high[1], high[2]], [low[0], high[1], high[2]],
    ])
    for left, right in (
        (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ):
        ax.plot(*zip(corners[left], corners[right]), color=color, linewidth=0.8)


def _draw_world(ax, scenario: ScenarioDefinition) -> None:
    world = np.asarray(scenario.world_size, dtype=float)
    _draw_box(ax, np.zeros(3), world, color="0.75")
    for obstacle in scenario.world.aabbs:
        _draw_box(ax, obstacle.low, obstacle.high, color="0.25")
    theta = np.linspace(0.0, 2.0 * np.pi, 48)
    for obstacle in scenario.world.cylinders:
        for z_value in (obstacle.z_low, obstacle.z_high):
            ax.plot(
                obstacle.center_xy[0] + obstacle.radius * np.cos(theta),
                obstacle.center_xy[1] + obstacle.radius * np.sin(theta),
                np.full_like(theta, z_value),
                color="0.25",
                linewidth=0.8,
            )
    ax.set_xlim(0.0, world[0])
    ax.set_ylim(0.0, world[1])
    ax.set_zlim(0.0, world[2])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.view_init(elev=28.0, azim=-58.0)


def _goal(record: dict[str, Any]) -> np.ndarray | None:
    value = record.get("goal_before", record.get("goal"))
    return None if value is None else np.asarray(value, dtype=float)


def _completed_goals(records: Iterable[dict[str, Any]]) -> np.ndarray:
    goals = [_goal(record) for record in records if record.get("task_completed_now")]
    goals = [goal for goal in goals if goal is not None]
    return np.empty((0, 3)) if not goals else np.stack(goals)


def _hud(record: dict[str, Any], station: np.ndarray) -> str:
    position = np.asarray(record["position"], dtype=float)
    goal = _goal(record)
    distance_goal = record.get("distance_to_goal_after")
    if distance_goal is None and goal is not None:
        distance_goal = float(np.linalg.norm(goal - position))
    distance_station = float(np.linalg.norm(station - position))
    return "\n".join((
        f"step: {record.get('step')}",
        f"task: {record.get('task_id_before', record.get('task_id'))}",
        f"tasks completed: {record.get('tasks_completed', 0)}",
        f"energy: {float(record.get('energy', np.nan)):.3f}",
        f"energy margin: {float(record.get('energy_margin', np.nan)):.3f}",
        f"mode: {record.get('persistent_mode')}",
        f"authority: {record.get('execution_authority')}",
        f"goal distance: {float(distance_goal):.3f}",
        f"station distance: {distance_station:.3f}",
    ))


def _draw_trajectory(ax, records: list[dict[str, Any]], scenario: ScenarioDefinition) -> None:
    _draw_world(ax, scenario)
    positions = np.asarray([record["position"] for record in records], dtype=float)
    start = np.asarray(records[0].get("position_before", positions[0]), dtype=float)
    current = positions[-1]
    goal = _goal(records[-1])
    completed = _completed_goals(records)
    station = np.asarray(scenario.station_position, dtype=float)
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color="#2667ff", linewidth=1.5, label="trajectory")
    ax.scatter(*start, color="#00a878", marker="o", s=45, label="start")
    ax.scatter(*current, color="#111111", marker="^", s=65, label="UAV")
    ax.scatter(*station, color="#ff8c00", marker="s", s=65, label="station")
    if goal is not None:
        ax.scatter(*goal, color="#d7263d", marker="*", s=110, label="current goal")
    if completed.size:
        ax.scatter(completed[:, 0], completed[:, 1], completed[:, 2], color="#8f2dff", marker="*", s=65, label="completed goals")
    velocity = np.asarray(records[-1].get("velocity", (0.0, 0.0, 0.0)), dtype=float)
    if np.linalg.norm(velocity) > 1e-9:
        ax.quiver(*current, *velocity, length=0.5, normalize=True, color="#111111")
    ax.legend(loc="upper right", fontsize=7)


def render_trajectory(
    records: list[dict[str, Any]],
    scenario: ScenarioDefinition,
    png_path: str | Path,
    gif_path: str | Path,
    *,
    frame_stride: int = 10,
    fps: int = 12,
) -> None:
    png_path = Path(png_path)
    gif_path = Path(gif_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(9.6, 6.4), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    _draw_trajectory(axis, records, scenario)
    axis.set_title("Persistent UAV held-out trajectory")
    figure.text(0.015, 0.72, _hud(records[-1], scenario.station_position), family="monospace", fontsize=8)
    figure.savefig(png_path, dpi=150)
    plt.close(figure)

    frame_indices = list(range(0, len(records), max(1, frame_stride)))
    if frame_indices[-1] != len(records) - 1:
        frame_indices.append(len(records) - 1)
    figure = plt.figure(figsize=(9.6, 6.4))
    axis = figure.add_subplot(111, projection="3d")
    hud = figure.text(0.015, 0.72, "", family="monospace", fontsize=8)

    def update(frame_number: int):
        index = frame_indices[frame_number]
        axis.clear()
        _draw_trajectory(axis, records[: index + 1], scenario)
        axis.set_title("Persistent UAV held-out trajectory")
        hud.set_text(_hud(records[index], scenario.station_position))
        return (hud,)

    movie = animation.FuncAnimation(figure, update, frames=len(frame_indices), interval=1000 / fps, blit=False)
    movie.save(gif_path, writer=animation.PillowWriter(fps=fps))
    plt.close(figure)
