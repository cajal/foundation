import numpy as np
from scipy.interpolate import interp1d
from .logging import logger


def fill_nans(trace):
    """Fills nans with linear interpolation

    Parameters
    ----------
    trace : 1D array
        values with nans

    Returns
    -------
    1D trace
        trace with interpolated nans
    """
    nan = np.isnan(trace)
    if nan.all():
        raise ValueError("Cannot fill when all values are nan.")

    out = trace.copy()
    out[nan] = np.interp(
        x=np.nonzero(nan)[0],
        xp=np.nonzero(~nan)[0],
        fp=trace[~nan],
    )
    return out


def monotonic(trace):
    """Determines if trace monotonically increases

    Parameters
    ----------
    trace : 1D array
        trace values

    Returns
    -------
    bool
        whether trace monotonically increases
    """
    delt = np.diff(trace)
    return bool(np.nanmin(delt) > 0)


def truncate(*traces, tolerance=1):
    """Truncates traces to the same length

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
    lengths = tuple(map(len, traces))
    min_len = min(lengths)
    max_len = max(lengths)

    if max_len - min_len > tolerance:
        raise ValueError(f"Arrays differ in length by more than {tolerance}")

    if max_len > min_len:
        logger.info(f"Truncating {max_len - min_len} frames")

    return tuple(trace[:min_len] for trace in traces)
