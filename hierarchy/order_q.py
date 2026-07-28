import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrderQNetwork(nn.Module):
    """Shared per-order Q network with fixed 13D absolute continuous input."""

    input_dim = 13

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, uav_state, order_features):
        x = torch.cat([uav_state, order_features], dim=-1)
        return self.net(x).squeeze(-1)


class OrderQTrainer:
    def __init__(self, q_net, lr=3e-4, gamma=0.99, target_update_tau=0.01, device="cpu"):
        self.q_net = q_net.to(device)
        self.target_q_net = copy.deepcopy(q_net).to(device)
        self.target_q_net.eval()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = float(gamma)
        self.target_update_tau = float(target_update_tau)
        self.device = torch.device(device)

    def _tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def train_step(self, batch):
        uav_state = self._tensor(batch["uav_state"])
        order = self._tensor(batch["order"])
        reward = self._tensor(batch["option_return"]).view(-1)
        duration = self._tensor(batch["duration"]).view(-1)
        done = self._tensor(batch["episode_done"]).view(-1)
        pred = self.q_net(uav_state, order)

        next_max = []
        with torch.no_grad():
            for idx, next_orders in enumerate(batch["next_feasible_orders"]):
                if len(next_orders) == 0:
                    next_max.append(torch.tensor(0.0, device=self.device))
                    continue
                next_order_tensor = self._tensor(next_orders)
                next_state = self._tensor(batch["next_uav_state"][idx]).view(1, -1)
                next_state = next_state.expand(next_order_tensor.size(0), -1)
                online_idx = torch.argmax(self.q_net(next_state, next_order_tensor))
                next_max.append(self.target_q_net(next_state, next_order_tensor)[online_idx])
            next_max = torch.stack(next_max)
            target = reward + torch.pow(
                torch.full_like(duration, self.gamma), duration
            ) * (1.0 - done) * next_max

        loss = F.huber_loss(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update_target()
        return {"order_q_loss": float(loss.detach().cpu().item())}

    def soft_update_target(self):
        with torch.no_grad():
            for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
                target_param.mul_(1.0 - self.target_update_tau).add_(
                    param, alpha=self.target_update_tau
                )

