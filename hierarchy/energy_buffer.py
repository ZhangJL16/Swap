import random
import pickle

import numpy as np


class EnergyReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.storage = []
        self.position = 0

    def __len__(self):
        return len(self.storage)

    def add(self, transition):
        transition = {
            "position": np.asarray(transition["position"], dtype=np.float32),
            "velocity": np.asarray(transition["velocity"], dtype=np.float32),
            "target": np.asarray(transition["target"], dtype=np.float32),
            "step_energy": float(transition["step_energy"]),
            "next_position": np.asarray(transition["next_position"], dtype=np.float32),
            "next_velocity": np.asarray(transition["next_velocity"], dtype=np.float32),
            "goal_done": float(transition["goal_done"]),
            "loaded_leg": float(transition["loaded_leg"]),
        }
        if len(self.storage) < self.capacity:
            self.storage.append(transition)
        else:
            self.storage[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.storage, min(int(batch_size), len(self.storage)))
        return {key: np.stack([item[key] for item in batch]) for key in batch[0]}

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
