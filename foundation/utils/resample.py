import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import windows


# ------- Trace Functions -------


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


def frame_index(time, period):
    """Returns the frame index that the time belongs to

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
    return frame_index(end - start, period) + 1


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


# ------- Trace Base -------


class Trace:
    """Trace Base"""

    def __init__(self, times, values, target_period):
        """
        Parameters
        -------
        times : 1D array
            trace times, monotonically increasing
        values : 1D array
            trace values, same length as times
        target_period : float
            target sampling period
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

        self.target_period = target_period
        self.source_period = np.nanmedian(np.diff(times))

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

    def __call__(self, start, end, offset=0):
        """
        Parameters
        ----------
        start : float
            start time
        end : float
            end time
        offset : float
            offset time

        Returns
        -------
        1D array | None
        """
        x = sample_times(
            start=self.transform_times(start) + offset,
            end=self.transform_times(end) + offset,
            period=self.target_period,
        )
        y = self.transform_values(
            values=self.interp(x),
            inverse=True,
        )
        return y


# ------- Trace Types -------


class Nans(Trace):
    """Trace Nans"""

    @property
    def x(self):
        return fill_nans(self.transform_times(self.times))

    def transform_values(self, values, inverse=False):
        if inverse:
            return values > 0
        else:
            nans = np.isnan(self.times) | np.isnan(values)
            return nans * 1.0


class Hamming(Trace):
    """Hamming filtered trace"""

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


class LowpassHamming(Trace):
    """Lowpass Hamming filtered trace"""

    def __init__(self, times, values, target_period, lowpass_period):
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
        """
        self.lowpass_period = lowpass_period

        super().__init__(times=times, values=values, target_period=target_period)

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

            print(self.lowpass_period, self.source_period, r)

        return y
