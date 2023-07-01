import io
import numpy as np


def torch_save(obj):
    """
    Parameters
    ----------
    obj : object
        object to save

    Returns
    -------
    1D array (np.uint8)
        byte array
    """
    import torch

    f = io.BytesIO()

    torch.save(obj, f)
    array = np.frombuffer(f.getvalue(), dtype=np.uint8)

    f.close()
    return array


def torch_load(array, map_location=None):
    """
    Parameters
    ----------
    array : 1D array (np.uint8)
        byte array to load from
    map_location : None | str | torch.device | Callable
        map location for tensors

    Returns
    -------
    object
        loaded object
    """
    import torch

    f = io.BytesIO(array.tobytes())

    obj = torch.load(f, map_location=map_location)

    f.close()
    return obj
