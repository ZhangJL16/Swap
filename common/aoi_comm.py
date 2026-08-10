import numpy as np


class AoIMessageEnhancer:
    """AoI-based receiver-side message filter.

    The sender transmits a masked observation. The receiver maintains the latest
    transmitted message and an AoI counter for each sender-receiver pair. When a
    new message is not received, the AoI increases by one slot. A simple
    paper-style filtering mask is then generated: the first Delta positions are
    zeroed and the rest are kept.
    """

    def __init__(self, n_agents, msg_dim, aoi_threshold=1.0, max_keep_dim=None):
        self.n_agents = int(n_agents)
        self.msg_dim = int(msg_dim)
        self.aoi_threshold = max(float(aoi_threshold), 1e-6)
        if max_keep_dim is None:
            self.max_keep_dim = self.msg_dim
        else:
            self.max_keep_dim = int(np.clip(int(max_keep_dim), 0, self.msg_dim))
        self.reset()

    def reset(self):
        self.cached_messages = np.zeros(
            (self.n_agents, self.n_agents, self.msg_dim), dtype=np.float32
        )
        self.age_steps = np.zeros((self.n_agents, self.n_agents), dtype=np.float32)
        self.last_mean_aoi = 0.0

    def _share_ratio(self, action_idx):
        keep_count = int(np.clip(int(action_idx), 0, self.max_keep_dim))
        share_ratio = keep_count / float(max(self.msg_dim, 1))
        return share_ratio, keep_count

    def _mapping_mask(self, raw_message, keep_count, priority_scores=None):
        mask = np.zeros(self.msg_dim, dtype=np.float32)
        if keep_count <= 0:
            return mask
        if keep_count >= self.msg_dim:
            mask.fill(1.0)
            return mask
        # Follow the paper-style communication matrix more closely:
        # for a chosen sharing ratio, preserve the leading observation entries.
        mask[:keep_count] = 1.0
        return mask

    def _aoi_mask(self, age_value):
        shift = int(np.clip(np.floor(age_value / self.aoi_threshold), 0, self.msg_dim))
        mask = np.ones(self.msg_dim, dtype=np.float32)
        if shift > 0:
            mask[:shift] = 0.0
        return mask

    def build_transmission(self, raw_message, action_idx, priority_scores=None):
        raw_message = np.asarray(raw_message, dtype=np.float32).reshape(-1)
        if raw_message.size > self.msg_dim:
            raw_message = raw_message[: self.msg_dim]
        elif raw_message.size < self.msg_dim:
            raw_message = np.pad(raw_message, (0, self.msg_dim - raw_message.size))
        share_ratio, keep_count = self._share_ratio(action_idx)
        mapping_mask = self._mapping_mask(raw_message, keep_count, priority_scores)
        if keep_count > 0:
            transmitted = raw_message * mapping_mask
            sent = 1.0
        else:
            transmitted = np.zeros(self.msg_dim, dtype=np.float32)
            sent = 0.0
        stats = {
            "share_ratio": float(share_ratio),
            "keep_count": int(keep_count),
            "sent": sent,
        }
        return transmitted.astype(np.float32), stats

    def receive_message(self, receiver_idx, sender_idx, transmitted_message, sent):
        transmitted_message = np.asarray(transmitted_message, dtype=np.float32).reshape(
            self.msg_dim
        )
        cached = self.cached_messages[receiver_idx, sender_idx]
        if float(sent) > 0.0:
            cached[:] = transmitted_message
            self.age_steps[receiver_idx, sender_idx] = 0.0
        else:
            self.age_steps[receiver_idx, sender_idx] += 1.0

        age_value = float(self.age_steps[receiver_idx, sender_idx])
        aoi_mask = self._aoi_mask(age_value)
        filtered = cached * aoi_mask

        self.last_mean_aoi = age_value
        stats = {
            "mean_aoi": age_value,
            "fresh_ratio": float(aoi_mask.mean()) if aoi_mask.size else 1.0,
        }
        return filtered.astype(np.float32), stats
