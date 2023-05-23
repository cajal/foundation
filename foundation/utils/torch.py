import io
import torch
import numpy as np


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
