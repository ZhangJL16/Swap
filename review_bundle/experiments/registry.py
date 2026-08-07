METHODS = ("sac", "penalty_sac", "shield_sac", "generator_sac")
SCENARIOS = ("mission_open", "mission_obstacle", "mission_narrow", "mission_energy_tight")


def validate_method(method: str) -> str:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    return method

