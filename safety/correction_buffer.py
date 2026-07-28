import random
import pickle

import numpy as np


class CorrectionBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.storage = []
        self.position = 0

    def __len__(self):
        return len(self.storage)

    def add(self, item):
        transition = {
            "safety_state_abs": np.asarray(item["safety_state_abs"], dtype=np.float32),
            "raw_joint_action": np.asarray(item["raw_joint_action"], dtype=np.float32),
            "correct_joint_action": np.asarray(item["correct_joint_action"], dtype=np.float32),
            "constraint_A": np.asarray(item["constraint_A"], dtype=np.float32),
            "constraint_c": np.asarray(item["constraint_c"], dtype=np.float32),
            "active_mask": np.asarray(item["active_mask"], dtype=np.float32),
        }
        if len(self.storage) < self.capacity:
            self.storage.append(transition)
        else:
            self.storage[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.storage, min(int(batch_size), len(self.storage)))
        return {key: [item[key] for item in batch] for key in batch[0]}

    def save(self, path):
        with open(path, "wb") as handle:
            pickle.dump(
                {
                    "capacity": self.capacity,
                    "storage": self.storage,
                    "position": self.position,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def load(self, path):
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        self.capacity = int(data.get("capacity", self.capacity))
        self.storage = list(data.get("storage", []))
        self.position = int(data.get("position", len(self.storage) % max(self.capacity, 1)))
