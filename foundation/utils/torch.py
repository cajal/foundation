import io
import os
import torch
import numpy as np
from contextlib import contextmanager


def save_to_array(obj):
    """
    Parameters
    ----------
    obj : object
        saved object

    Returns
    -------
    1D array (np.uint8)
        byte array
    """
    f = io.BytesIO()

    torch.save(obj, f)
    array = np.frombuffer(f.getvalue(), dtype=np.uint8)

    f.close()
    return array


def load_from_array(array, map_location=None):
    """
    Parameters
    ----------
    array : 1D array (np.uint8)
        byte array
    map_location : None | str | torch.device | Callable
        map location for tensors

    Returns
    -------
    object
        loaded object
    """
    f = io.BytesIO(array.tobytes())

    obj = torch.load(f, map_location=map_location)

    f.close()
    return obj


@contextmanager
def use_cuda(device=None):
    """Context manager that explicitly enables cuda usage

    Parameters
    ----------
    device : None | int
        cuda device
    """
    assert torch.cuda.is_available()

    if device is None:
        device = torch.cuda.current_device()
    else:
        device = int(device)
        assert 0 <= device < torch.cuda.device_count()

    prev = os.getenv("FOUNDATION_CUDA", "-1")
    os.environ["FOUNDATION_CUDA"] = str(device)
    try:
        with torch.cuda.device(device):
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
    if not torch.cuda.is_available():
        return False

    cuda = os.getenv("FOUNDATION_CUDA", "-1")
    return int(cuda) >= 0
