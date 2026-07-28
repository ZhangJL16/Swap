import torch
import torch.nn as nn


class DistilledCorrector(nn.Module):
    def __init__(self, safety_state_dim, joint_action_dim, hidden_dim=256):
        super().__init__()
        self.safety_state_dim = int(safety_state_dim)
        self.joint_action_dim = int(joint_action_dim)
        self.net = nn.Sequential(
            nn.Linear(self.safety_state_dim + self.joint_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.joint_action_dim),
        )

    def forward(self, safety_state_abs, raw_joint_action):
        return self.net(torch.cat([safety_state_abs, raw_joint_action], dim=-1))

