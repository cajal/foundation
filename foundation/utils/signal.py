from scipy.signal import windows


def lowpass_filter(source_period, target_period, filter_type="hamming"):
    """
    Parameters
    ----------
    source_period : float
        source sampling period
    target_period : float
        target sampling period
    filter_type : str
        lowpass filter type

    Returns
    -------
    np.array
        1D lowpass filter
    """
    if target_period < source_period:
        raise ValueError("target_period must be >= source_period")

    if filter_type == "hamming":
        ratio = round(target_period / source_period * 10) // 10
        h = windows.hamming(ratio * 2 + 1)
        return h / h.sum()

    else:
        raise NotImplementedError(f"Filter type '{filter_type}' not implemented")
