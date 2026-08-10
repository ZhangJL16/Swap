import os
import random
from contextlib import contextmanager

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    import torch
except ImportError:  # pragma: no cover - torch is a project dependency.
    torch = None


MAX_NUMPY_SEED = 2**32 - 1


def normalize_seed(seed):
    return int(seed) % MAX_NUMPY_SEED


def seed_everything(seed, deterministic_torch=True):
    seed = normalize_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is None:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            pass
        except Exception:
            pass

    return seed


def derive_episode_seed(args, evaluate, episode_index):
    episode_index = int(0 if episode_index is None else episode_index)
    stride = int(getattr(args, "episode_seed_stride", 1))
    if evaluate:
        base_seed = int(getattr(args, "eval_seed", int(getattr(args, "seed", 123)) + 100000))
    else:
        base_seed = int(getattr(args, "seed", 123))
    return normalize_seed(base_seed + episode_index * stride)


@contextmanager
def preserve_rng_state(include_torch=False):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = None
    cuda_states = None
    if include_torch and torch is not None:
        torch_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            cuda_states = torch.cuda.get_rng_state_all()

    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if include_torch and torch is not None and torch_state is not None:
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_states)


@contextmanager
def temporary_seed(seed, include_torch=True):
    with preserve_rng_state(include_torch=include_torch):
        seed_everything(seed)
        yield


def reset_env_with_seed(env, seed):
    if seed is None:
        return env.reset()
    try:
        return env.reset(seed=int(seed))
    except TypeError:
        return env.reset()
