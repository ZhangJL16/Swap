import torch
import torch.nn as nn


class ConstrainedFlow(nn.Module):
    def __init__(self, safety_state_dim, joint_action_dim, hidden_dim=256):
        super().__init__()
        self.safety_state_dim = int(safety_state_dim)
        self.joint_action_dim = int(joint_action_dim)
        input_dim = self.safety_state_dim + 2 * self.joint_action_dim + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.joint_action_dim),
        )

    def forward(self, safety_state_abs, raw_joint_action, current_joint_action, flow_time):
        if flow_time.dim() == 1:
            flow_time = flow_time.unsqueeze(-1)
        x = torch.cat(
            [safety_state_abs, raw_joint_action, current_joint_action, flow_time],
            dim=-1,
        )
        return self.net(x)

    def integrate(self, safety_state_abs, raw_joint_action, steps):
        action = raw_joint_action
        steps = int(steps)
        dt = 1.0 / max(steps, 1)
        for idx in range(steps):
            tau = torch.full(
                (raw_joint_action.size(0), 1),
                float(idx) / max(steps, 1),
                dtype=raw_joint_action.dtype,
                device=raw_joint_action.device,
            )
            velocity = self(safety_state_abs, raw_joint_action, action, tau)
            action = action + dt * velocity
        return action

