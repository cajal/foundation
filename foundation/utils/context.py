import os
from contextlib import contextmanager


@contextmanager
def torch_rng(seed=None):
    """Context manager that forks the torch RNG and optionally sets a manual seed in the forked state

    Parameters
    ----------
    seed : int | None
        seed for generating random numbers
    """
    from torch import cuda, random

    devices = list(range(cuda.device_count()))

    try:
        with random.fork_rng(devices):
            if seed is not None:
                random.manual_seed(seed)
            yield
    finally:
        pass


@contextmanager
def use_cuda(device=None):
    """Context manager that explicitly enables cuda usage

    Parameters
    ----------
    device : None | int
        cuda device
    """
    from torch import cuda

    assert cuda.is_available()

    if device is None:
        device = cuda.current_device()
    else:
        device = int(device)
        assert 0 <= device < cuda.device_count()

    prev = os.getenv("FOUNDATION_CUDA", "-1")
    os.environ["FOUNDATION_CUDA"] = str(device)

    try:
        with cuda.device(device):
            yield
    finally:
        os.environ["FOUNDATION_CUDA"] = prev


def cuda_enabled():
    """Check if cuda is explictly enabled

    Returns
    -------
    bool
        whether cuda usage is explicitly enabled
    """
    from torch import cuda

    if not cuda.is_available():
        return False

    env = os.getenv("FOUNDATION_CUDA", "-1")
    return int(env) >= 0
