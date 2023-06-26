import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import windows


# ------- Trace Functions -------


def truncate(*traces, tolerance=1):
    """Truncates traces to the same length

    Parameters
    ----------
    *traces : 1D array
        traces to truncate
    tolerance : int
        length mismatch tolerance

    Returns
    -------
    tuple[1D array]
        traces truncated to the same length
    """
    lengths = list(map(len, traces))
    max_len = max(lengths)
    min_len = min(lengths)

    if max_len - min_len > tolerance:
        raise ValueError(f"Traces differ in length by more than {tolerance}")

    return tuple(trace[:min_len] for trace in traces)


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


def target_index(time, period):
    """Target index for the provided time and sampling period

    Parameters
    ----------
    time : array-like
        time
    period : float
        sampling period

    Returns
    -------
    array-like
        dtype = int
    """
    index = np.round(time / period, 1)
    return np.floor(index).astype(int)


def flip_index(times, period):
    """Interpolated flip index for the provided flip times and sampling period

    Parameters
    ----------
    times : 1D array
        flip times
    period : float
        sampling period

    Returns
    -------
    1D array
        dtype = int
    """
    assert monotonic(times)
    assert times[0] == 0

    index = target_index(times, period)
    samples = np.arange(index[-1] + 1)

    new = np.diff(index, prepend=-1) > 0
    previous = interp1d(x=index[new], y=np.where(new)[0], kind="previous")

    return previous(samples).astype(int)


def samples(start, end, period):
    """Number of samples

    Parameters
    ----------
    start : float
        start time
    end : float
        end time
    period : float
        sampling period

    Returns
    -------
    int
        number of samples
    """
    return target_index(end - start, period) + 1


def sample_times(start, end, period):
    """Sampling times

    Parameters
    ----------
    start : float
        start time
    end : float
        end time
    period : float
        sampling period

    Returns
    -------
    1D array
        sampling times
    """
    n = samples(start, end, period)
    return np.arange(n) * period + start


# ------- Resample Base -------


class Resample:
    """Resample Base"""

    def __init__(self, times, values, target_period, target_offset=0):
        """
        Parameters
        -------
        times : 1D array
            trace times, monotonically increasing
        values : 1D array
            trace values, same length as times
        target_period : float
            target sampling period
        target_offset : float
            target sampling offset
        """
        if not times.ndim == values.ndim == 1:
            raise ValueError("Times and Values must be 1D")

        if times.size != values.size:
            raise ValueError("Times and Values are not the same size")

        if not monotonic(times):
            raise ValueError("Times do not monotonically increase.")

        self.times = times
        self.values = values

        self.median_time = np.nanmedian(times)
        self.median_value = np.nanmedian(values)

        self.source_period = np.nanmedian(np.diff(times))
        self.target_period = target_period
        self.target_offset = target_offset

        self.interp = interp1d(
            x=self.x,
            y=self.y,
            kind=self.kind,
            bounds_error=False,
            fill_value=np.nan,
        )

    @property
    def x(self):
        return self.transform_times(self.times)

    @property
    def y(self):
        return self.transform_values(self.values)

    @property
    def kind(self):
        return "linear"

    @property
    def dtype(self):
        return np.float32

    def transform_times(self, times, inverse=False):
        if inverse:
            return times + self.median_time
        else:
            return times - self.median_time

    def transform_values(self, values, inverse=False):
        if inverse:
            return values + self.median_value
        else:
            return values - self.median_value

    def __call__(self, start, end):
        """
        Parameters
        ----------
        start : float
            start time
        end : float
            end time

        Returns
        -------
        1D array | None
        """
        x = sample_times(
            start=self.transform_times(start),
            end=self.transform_times(end),
            period=self.target_period,
        )
        x = x + self.target_offset
        y = self.transform_values(
            values=self.interp(x),
            inverse=True,
        )
        return y.astype(self.dtype)


# ------- Resample Types -------


class Nans(Resample):
    """Trace Nans"""

    @property
    def x(self):
        return fill_nans(self.transform_times(self.times))

    @property
    def dtype(self):
        return bool

    def transform_values(self, values, inverse=False):
        if inverse:
            return values > 0
        else:
            nans = np.isnan(self.times) | np.isnan(values)
            return nans * 1.0


class Hamming(Resample):
    """Resampling with Hamming filtering"""

    @property
    def x(self):
        return fill_nans(self.transform_times(self.times))

    @property
    def y(self):
        y = fill_nans(self.transform_values(self.values))

        if self.target_period > self.source_period:
            r = round(self.target_period / self.source_period)
            h = windows.hamming(r * 2 + 1)
            f = h / h.sum()
            y = np.convolve(y, f, mode="same")

        return y


class LowpassHamming(Resample):
    """Resample with Lowpass Hamming filtering"""

    def __init__(self, times, values, target_period, lowpass_period, target_offset=0):
        """
        Parameters
        -------
        times : 1D array
            trace times, monotonically increasing
        values : 1D array
            trace values, same length as times
        target_period : float
            target sampling period
        lowpass_period : float
            lowpass filter period
        target_offset : float
            target sampling offset
        """
        self.lowpass_period = lowpass_period

        super().__init__(times=times, values=values, target_period=target_period, target_offset=target_offset)

    @property
    def x(self):
        return fill_nans(self.transform_times(self.times))

    @property
    def y(self):
        y = fill_nans(self.transform_values(self.values))

        if self.lowpass_period > self.source_period:
            r = round(self.lowpass_period / self.source_period)
            h = windows.hamming(r * 2 + 1)
            f = h / h.sum()
            y = np.convolve(y, f, mode="same")

        return y
