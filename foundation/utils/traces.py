import numpy as np
from .signal import lowpass_filter


def lowpass(trace, source_period, target_period, filter_type="hamming"):
    """
    Parameters
    ----------
    trace : 1D array
        trace values
    source_period : float
        source sampling period
    target_period : float
        target sampling period
    filter_type : str
        lowpass filter type

    Returns
    -------
    1D array
        lowpassed trace
    """
    filt = lowpass_filter(
        source_period=source_period,
        target_period=target_period,
        filter_type=filter_type,
    )
    return np.convolve(trace, filt, mode="same")


def consecutive_nans(trace):
    """
    Parameters
    ----------
    trace : 1D array
        trace values

    Returns
    -------
    1D array
        consecutive NaNs
    """
    nan = np.isnan(trace)
    pad = np.concatenate([[0], nan, [0]])
    delt = np.nonzero(np.diff(pad))[0]
    nans = np.zeros_like(trace, dtype=int)

    for a, b in delt.reshape(-1, 2):
        nans[a:b] = b - a

    return nans


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
    Tuple[array]
        1D arrays of the same length
    """
    lengths = list(map(len, traces))
    min_len = min(lengths)
    max_len = max(lengths)

    if max_len - min_len > tolerance:
        raise ValueError(f"Traces differ in length by more than {tolerance}")

    if max_len > min_len:
        logger.info(f"Truncating {max_len - min_len} frames")

    return tuple(trace[:min_len] for trace in traces)
