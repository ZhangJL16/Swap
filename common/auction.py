import numpy as np


def auction_assign_min_cost(cost_matrix, epsilon=1e-3):
    """Assign rows to columns with a Bertsekas-style auction.

    Parameters
    ----------
    cost_matrix : array_like
        Shape ``(n_agents, n_items)``. Lower cost is better.
    epsilon : float
        Small bid increment used to break ties.

    Returns
    -------
    dict
        Mapping from row index to column index.
    """
    costs = np.asarray(cost_matrix, dtype=np.float32)
    if costs.ndim != 2:
        raise ValueError("cost_matrix must be two-dimensional")
    n_agents, n_real_items = costs.shape
    if n_agents == 0 or n_real_items == 0:
        return {}
    if n_agents > n_real_items:
        dummy_items = n_agents - n_real_items
        costs = np.concatenate(
            [costs, np.zeros((n_agents, dummy_items), dtype=np.float32)],
            axis=1,
        )
    n_items = costs.shape[1]

    values = -costs
    prices = np.zeros(n_items, dtype=np.float32)
    item_owner = np.full(n_items, -1, dtype=np.int64)
    agent_item = np.full(n_agents, -1, dtype=np.int64)
    unassigned = list(range(n_agents))

    while unassigned:
        agent_idx = unassigned.pop(0)
        utilities = values[agent_idx] - prices
        best_item = int(np.argmax(utilities))
        best_utility = float(utilities[best_item])
        if n_items > 1:
            utilities_without_best = utilities.copy()
            utilities_without_best[best_item] = -np.inf
            second_utility = float(np.max(utilities_without_best))
        else:
            second_utility = best_utility - float(epsilon)

        bid = best_utility - second_utility + float(epsilon)
        previous_owner = int(item_owner[best_item])
        prices[best_item] += bid
        item_owner[best_item] = agent_idx
        agent_item[agent_idx] = best_item

        if previous_owner >= 0:
            agent_item[previous_owner] = -1
            unassigned.append(previous_owner)

    return {
        int(agent_idx): int(item_idx)
        for agent_idx, item_idx in enumerate(agent_item)
        if 0 <= item_idx < n_real_items
    }
