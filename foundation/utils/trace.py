import numpy as np
from scipy.interpolate import interp1d
from .errors import OutOfBounds
from .logging import logger


def fill_nans(array):
    """Fills nans with linear interpolation

    Parameters
    ----------
    array : 1D array
        values with nans

    Returns
    -------
    1D array
        array of same length with interpolated nans
    """
    nan = np.isnan(array)
    if nan.all():
        raise ValueError("Cannot fill when all values are nan.")

    out = array.copy()
    out[nan] = np.interp(
        x=np.nonzero(nan)[0],
        xp=np.nonzero(~nan)[0],
        fp=array[~nan],
    )
    return out


def truncate(*arrays, tolerance=1):
    """Truncates arrays to the same length

    Parameters
    ----------
    traces : Tuple[1D array]
        traces that possibly differ in length
    tolerance : int
        tolerance for length mismatches

    Returns
    -------
    Tuple[1D array]
        traces of the same length
    """
    lengths = tuple(map(len, arrays))
    min_len = min(lengths)
    max_len = max(lengths)

    if max_len - min_len > tolerance:
        raise ValueError(f"Arrays differ in length by more than {tolerance}")

    if max_len > min_len:
        logger.info(f"Truncating {max_len - min_len} frames")

    return tuple(array[:min_len] for array in arrays)
