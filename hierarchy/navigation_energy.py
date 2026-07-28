import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class NavigationEnergyNetwork(nn.Module):
    """Goal-conditioned navigation energy model.

    Input is strictly continuous absolute state: position, velocity and target.
    Loaded/empty selection is handled outside the network by choosing the head.
    """

    input_dim = 9

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.empty_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())
        self.loaded_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())

    def forward(self, position, velocity, target, loaded_leg=False):
        x = torch.cat([position, velocity, target], dim=-1)
        features = self.trunk(x)
        if isinstance(loaded_leg, torch.Tensor):
            empty = self.empty_head(features)
            loaded = self.loaded_head(features)
            selector = loaded_leg.float().view(-1, 1)
            return torch.where(selector > 0.5, loaded, empty)
        return self.loaded_head(features) if loaded_leg else self.empty_head(features)

    def both_heads(self, position, velocity, target):
        x = torch.cat([position, velocity, target], dim=-1)
        features = self.trunk(x)
        return self.empty_head(features), self.loaded_head(features)


class NavigationEnergyTrainer:
    def __init__(self, model, lr=3e-4, gamma=0.99, target_update_tau=0.01, device="cpu"):
        self.model = model.to(device)
        self.target = copy.deepcopy(model).to(device)
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = float(gamma)
        self.target_update_tau = float(target_update_tau)
        self.device = torch.device(device)

    def _tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def train_step(self, batch):
        position = self._tensor(batch["position"])
        velocity = self._tensor(batch["velocity"])
        target = self._tensor(batch["target"])
        step_energy = self._tensor(batch["step_energy"]).view(-1, 1)
        next_position = self._tensor(batch["next_position"])
        next_velocity = self._tensor(batch["next_velocity"])
        done = self._tensor(batch["goal_done"]).view(-1, 1)
        loaded_leg = self._tensor(batch["loaded_leg"]).view(-1, 1)

        pred = self.model(position, velocity, target, loaded_leg=loaded_leg)
        with torch.no_grad():
            bootstrap = self.target(next_position, next_velocity, target, loaded_leg=loaded_leg)
            y = step_energy + self.gamma * (1.0 - done) * bootstrap
        loss = F.huber_loss(pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update_target()
        return {"energy_loss": float(loss.detach().cpu().item())}

    def soft_update_target(self):
        with torch.no_grad():
            for target_param, param in zip(self.target.parameters(), self.model.parameters()):
                target_param.mul_(1.0 - self.target_update_tau).add_(
                    param, alpha=self.target_update_tau
                )

