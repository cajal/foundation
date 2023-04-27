import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import windows
from .signal import lowpass_filter
from .logging import logger


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


# ------- Trace Base -------


class Trace:
    """Trace Base"""

    def __init__(self, times, values, target_period, dtype=np.float32):
        """
        Parameters
        -------
        times : 1D array
            trace times, monotonically increasing
        values : 1D array
            trace values, same length as times
        target_period : float
            target sampling period
        dtype : data-type
            output data type
        """
        if not times.ndim == values.ndim == 1:
            raise ValueError("Times and Values must be 1D")

        if times.size != values.size:
            raise ValueError("Times and Values are not the same size")

        if not monotonic(times):
            raise ValueError("Times do not monotonically increase.")

        self.times = times
        self.values = values
        self.dtype = dtype

        self.median_time = np.nanmedian(times)
        self.median_value = np.nanmedian(values)

        self.target_period = target_period
        self.source_period = np.nanmedian(np.diff(times))

        self.init()

        self.interp = interp1d(
            x=self.x,
            y=self.y,
            kind=self.kind,
            bounds_error=False,
            fill_value=np.nan,
        )

    def init(self):
        pass

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

    def __call__(self, start, samples):
        """
        Parameters
        ----------
        start : float
            start time
        samples : int
            number of samples

        Returns
        -------
        1D array | None
        """
        x = self.transform_times(start) + self.target_period * np.arange(samples)
        y = self.transform_values(self.interp(x), inverse=True)

        if np.isnan(y).any():
            return
        else:
            return y.astype(self.dtype)


# ------- Trace Types -------


class Nans(Trace):
    """Trace Nans"""

    @property
    def x(self):
        return fill_nans(self.transform_times(self.times))

    @property
    def y(self):
        nans = np.isnan(self.times) | np.isnan(self.values)
        y = np.zeros_like(self.values)
        y[nans] = 1
        return y

    def transform_values(self, values, inverse=False):
        return values

    def __call__(self, start, samples):
        samples = super().__call__(start, samples)

        if samples is None:
            return
        else:
            return samples > 0


class Hamming(Trace):
    """Hamming filtered trace"""

    def init(self):
        nans = np.isnan(self.times) | np.isnan(self.times)

        self._x = self.transform_times(self.times[~nans])
        self._y = self.transform_values(self.values[~nans])

        if self.target_period > self.source_period:
            logger.info("Target period is greater than source period. Filtering trace with Hamming window.")

            r = round(self.target_period / self.source_period)
            h = windows.hamming(r * 2 + 1)
            f = h / h.sum()

            self._y = np.convolve(self._y, f, mode="same")

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y
