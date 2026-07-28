import numpy as np
import torch
import torch.nn.functional as F


class FlowTrainer:
    def __init__(
        self,
        flow_net,
        lr=3e-4,
        flow_train_steps=8,
        cbf_coef=1.0,
        endpoint_coef=1.0,
        terminal_coef=1.0,
        flow_margin=0.0,
        kappa=1.0,
        device="cpu",
    ):
        self.flow_net = flow_net.to(device)
        self.optimizer = torch.optim.Adam(self.flow_net.parameters(), lr=lr)
        self.flow_train_steps = int(flow_train_steps)
        self.cbf_coef = float(cbf_coef)
        self.endpoint_coef = float(endpoint_coef)
        self.terminal_coef = float(terminal_coef)
        self.flow_margin = float(flow_margin)
        self.kappa = float(kappa)
        self.device = torch.device(device)

    def _tensor(self, value):
        if isinstance(value, list):
            value = np.stack(value)
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _constraint_tensors(self, batch):
        if not isinstance(batch["constraint_A"], list):
            return self._tensor(batch["constraint_A"]), self._tensor(batch["constraint_c"])
        raw_action_dim = self._tensor(batch["raw_joint_action"]).shape[-1]
        max_rows = max(max(np.asarray(item).shape[0], 1) for item in batch["constraint_A"])
        A = torch.zeros(
            len(batch["constraint_A"]),
            max_rows,
            raw_action_dim,
            dtype=torch.float32,
            device=self.device,
        )
        c = torch.full(
            (len(batch["constraint_c"]), max_rows),
            1e6,
            dtype=torch.float32,
            device=self.device,
        )
        for idx, (A_item, c_item) in enumerate(zip(batch["constraint_A"], batch["constraint_c"])):
            A_item = np.asarray(A_item, dtype=np.float32)
            c_item = np.asarray(c_item, dtype=np.float32).reshape(-1)
            rows = A_item.shape[0]
            if rows <= 0:
                continue
            A[idx, :rows] = self._tensor(A_item)
            c[idx, :rows] = self._tensor(c_item)
        return A, c

    def losses(self, batch):
        state = self._tensor(batch["safety_state_abs"])
        raw = self._tensor(batch["raw_joint_action"])
        correct = self._tensor(batch["correct_joint_action"])
        A, c = self._constraint_tensors(batch)
        batch_size = raw.size(0)

        tau = torch.rand(batch_size, 1, dtype=raw.dtype, device=self.device)
        a_tau = (1.0 - tau) * raw + tau * correct
        target_velocity = correct - raw
        pred_velocity = self.flow_net(state, raw, a_tau, tau)
        fm_loss = F.mse_loss(pred_velocity, target_velocity)

        margin_tau = torch.bmm(A, a_tau.unsqueeze(-1)).squeeze(-1) + c
        margin_rate = torch.bmm(A, pred_velocity.unsqueeze(-1)).squeeze(-1)
        required = self.kappa * torch.relu(self.flow_margin - margin_tau)
        cbf_loss = torch.relu(required - margin_rate).pow(2).mean()

        end = self.flow_net.integrate(state, raw, self.flow_train_steps)
        endpoint_loss = F.mse_loss(end, correct)
        terminal_margin = torch.bmm(A, end.unsqueeze(-1)).squeeze(-1) + c
        terminal_loss = torch.relu(self.flow_margin - terminal_margin).pow(2).mean()
        total = (
            fm_loss
            + self.cbf_coef * cbf_loss
            + self.endpoint_coef * endpoint_loss
            + self.terminal_coef * terminal_loss
        )
        return {
            "loss": total,
            "fm_loss": fm_loss,
            "cbf_loss": cbf_loss,
            "endpoint_loss": endpoint_loss,
            "terminal_loss": terminal_loss,
        }

    def train_step(self, batch):
        losses = self.losses(batch)
        self.optimizer.zero_grad()
        losses["loss"].backward()
        self.optimizer.step()
        return {key: float(value.detach().cpu().item()) for key, value in losses.items()}
