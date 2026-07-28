import random
import pickle

import numpy as np


class OrderReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.storage = []
        self.position = 0

    def __len__(self):
        return len(self.storage)

    def add(self, transition):
        item = {
            "uav_state": np.asarray(transition["uav_state"], dtype=np.float32),
            "order": np.asarray(transition["order"], dtype=np.float32),
            "option_return": float(transition["option_return"]),
            "duration": float(transition["duration"]),
            "next_uav_state": np.asarray(transition["next_uav_state"], dtype=np.float32),
            "next_feasible_orders": [
                np.asarray(order, dtype=np.float32)
                for order in transition.get("next_feasible_orders", [])
            ],
            "episode_done": float(transition["episode_done"]),
        }
        if len(self.storage) < self.capacity:
            self.storage.append(item)
        else:
            self.storage[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        samples = random.sample(self.storage, min(int(batch_size), len(self.storage)))
        return {
            "uav_state": np.stack([item["uav_state"] for item in samples]),
            "order": np.stack([item["order"] for item in samples]),
            "option_return": np.asarray([item["option_return"] for item in samples], dtype=np.float32),
            "duration": np.asarray([item["duration"] for item in samples], dtype=np.float32),
            "next_uav_state": np.stack([item["next_uav_state"] for item in samples]),
            "next_feasible_orders": [item["next_feasible_orders"] for item in samples],
            "episode_done": np.asarray([item["episode_done"] for item in samples], dtype=np.float32),
        }

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
