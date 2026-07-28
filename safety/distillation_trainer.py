import numpy as np
import torch
import torch.nn.functional as F


class DistillationTrainer:
    def __init__(
        self,
        student,
        lr=3e-4,
        teacher_coef=1.0,
        safety_coef=1.0,
        margin=0.0,
        device="cpu",
    ):
        self.student = student.to(device)
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=lr)
        self.teacher_coef = float(teacher_coef)
        self.safety_coef = float(safety_coef)
        self.margin = float(margin)
        self.device = torch.device(device)
        self.forward_calls = 0

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

    def train_step(self, batch):
        state = self._tensor(batch["safety_state_abs"])
        raw = self._tensor(batch["raw_joint_action"])
        flow_action = self._tensor(batch["flow_joint_action"])
        correct = self._tensor(batch["correct_joint_action"])
        A, c = self._constraint_tensors(batch)
        student_action = self.student(state, raw)
        self.forward_calls += 1
        flow_loss = F.mse_loss(student_action, flow_action)
        teacher_loss = F.mse_loss(student_action, correct)
        margin = torch.bmm(A, student_action.unsqueeze(-1)).squeeze(-1) + c
        safety_loss = torch.relu(self.margin - margin).pow(2).mean()
        loss = flow_loss + self.teacher_coef * teacher_loss + self.safety_coef * safety_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "student_loss": float(loss.detach().cpu().item()),
            "student_flow_loss": float(flow_loss.detach().cpu().item()),
            "student_teacher_loss": float(teacher_loss.detach().cpu().item()),
            "student_safety_loss": float(safety_loss.detach().cpu().item()),
        }
