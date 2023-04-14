import numpy as np
from .signal import lowpass_filter


def lowpass(trace, source_period, target_period, filter_type="hamming"):
    """
    Parameters
    ----------
    trace : np.array
        1D array
    source_period : float
        source sampling period
    target_period : float
        target sampling period
    filter_type : str
        lowpass filter type

    Returns
    -------
    np.array
        lowpassed 1D array
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
    trace : np.array
        1D array

    Returns
    -------
    np.array
        1D array indicating numbers of consecutive NaNs
    """
    nan = np.isnan(trace)
    pad = np.concatenate([[0], nan, [0]])
    delt = np.nonzero(np.diff(pad))[0]
    consec = np.zeros_like(trace, dtype=int)

    for a, b in delt.reshape(-1, 2):
        consec[a:b] = b - a

    return consec
