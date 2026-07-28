import numpy as np
import torch


def order_features(order, default_time_limit):
    return np.concatenate(
        [
            np.asarray(order.pickup_pos, dtype=np.float32),
            np.asarray(order.dropoff_pos, dtype=np.float32),
            np.asarray([float(getattr(order, "time_limit", default_time_limit))], dtype=np.float32),
        ]
    )


class HighLevelController:
    def __init__(self, order_q_net=None, energy_model=None, reserve=0.0, device="cpu"):
        self.order_q_net = order_q_net
        self.energy_model = energy_model
        self.reserve = float(reserve)
        self.device = torch.device(device)

    @staticmethod
    def uav_state(agent):
        return np.concatenate([agent.pos.astype(np.float32), agent.vel.astype(np.float32)])

    def estimate_order_energy(self, agent, order, station_positions):
        if self.energy_model is None:
            empty_to_pickup = float(np.linalg.norm(order.pickup_pos - agent.pos))
            loaded_to_dropoff = float(np.linalg.norm(order.dropoff_pos - order.pickup_pos))
            station = min(
                station_positions,
                key=lambda pos: float(np.linalg.norm(order.dropoff_pos - pos)),
            )
            return empty_to_pickup + loaded_to_dropoff + float(np.linalg.norm(order.dropoff_pos - station))
        with torch.no_grad():
            pos = torch.as_tensor(agent.pos, dtype=torch.float32, device=self.device).view(1, -1)
            vel = torch.as_tensor(agent.vel, dtype=torch.float32, device=self.device).view(1, -1)
            pickup = torch.as_tensor(order.pickup_pos, dtype=torch.float32, device=self.device).view(1, -1)
            drop = torch.as_tensor(order.dropoff_pos, dtype=torch.float32, device=self.device).view(1, -1)
            zero = torch.zeros_like(vel)
            e1 = self.energy_model(pos, vel, pickup, loaded_leg=False)
            e2 = self.energy_model(pickup, zero, drop, loaded_leg=True)
            e3_values = []
            for station in station_positions:
                station_t = torch.as_tensor(station, dtype=torch.float32, device=self.device).view(1, -1)
                e3_values.append(self.energy_model(drop, zero, station_t, loaded_leg=False))
            e3 = torch.min(torch.stack(e3_values))
            return float((e1 + e2 + e3).item())

    def feasible_orders(self, agent, orders, station_positions):
        feasible = []
        for order in orders:
            predicted = self.estimate_order_energy(agent, order, station_positions)
            if agent.energy >= predicted + self.reserve:
                feasible.append(order)
        return feasible

    def select_order(self, agent, orders, station_positions, default_time_limit):
        feasible = self.feasible_orders(agent, orders, station_positions)
        if not feasible:
            return None
        if self.order_q_net is None:
            return feasible[0]
        with torch.no_grad():
            uav = torch.as_tensor(self.uav_state(agent), dtype=torch.float32, device=self.device)
            features = np.stack([order_features(order, default_time_limit) for order in feasible])
            order_tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device)
            uav_tensor = uav.view(1, -1).expand(order_tensor.size(0), -1)
            q_values = self.order_q_net(uav_tensor, order_tensor)
            return feasible[int(torch.argmax(q_values).item())]

    @staticmethod
    def select_min_energy_station(agent, station_positions):
        return int(
            np.argmin(
                [
                    float(np.linalg.norm(np.asarray(station, dtype=np.float32) - agent.pos))
                    for station in station_positions
                ]
            )
        )

