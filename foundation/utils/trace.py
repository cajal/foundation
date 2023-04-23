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


# ------- Trace Gap -------


class Gap:
    """Nan time gap"""

    def __init__(self, times, values):
        """
        Parameters
        -------
        times : 1D array
            trace times, monotonically increasing
        values : 1D array
            trace values, same length as times
        """
        if not monotonic(times):
            raise ValueError("Times do not monotonically increase.")

        nans = np.isnan(times) | np.isnan(values)
        idx_nan = np.nonzero(nans)[0]
        idx_val = np.nonzero(~nans)[0]

        self.center = np.median(times[idx_val])
        self.ctimes = times - self.center
        self.index = interp1d(
            x=fill_nans(self.ctimes),
            y=np.arange(self.ctimes.size),
            kind="nearest",
            bounds_error=False,
            fill_value=np.nan,
        )
        left = interp1d(
            x=idx_val,
            y=self.ctimes[idx_val],
            kind="previous",
            bounds_error=False,
            fill_value=np.nan,
        )
        right = interp1d(
            x=idx_val,
            y=self.ctimes[idx_val],
            kind="next",
            bounds_error=False,
            fill_value=np.nan,
        )
        self.gaps = np.zeros_like(self.ctimes)
        self.gaps[idx_nan] = right(idx_nan) - left(idx_nan)

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
        float | None
            maximum gap in time between nans in trace times or values
                or
            None if 'start' or 'end' is out of bounds
        """
        bounds = [
            start - self.center,
            end - self.center,
        ]
        bounds = self.index(bounds)

        if np.isnan(bounds).any():
            return
        else:
            i, j = map(int, bounds)
            gap = self.gaps[i : j + 1].max()
            return float(gap)


# ------- Trace Resampling -------


class Resample:
    """Trace Resampler"""

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
        if not monotonic(times):
            raise ValueError("Times do not monotonically increase.")

        self.times = times
        self.values = values
        self.target_period = target_period
        self.source_period = np.nanmedian(np.diff(times))

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
        raise NotImplementedError()


class HammingResample(Resample):
    """Resample a hamming filtered trace"""

    def __init__(self, times, values, target_period):
        super().__init__(times, values, target_period)

        # fill nans in trace values
        v = fill_nans(self.values)

        # filter trace values hamming window
        if self.target_period > self.source_period:
            logger.info("Target period is greater than source period. Filtering trace with Hamming window.")
            r = round(self.target_period / self.source_period * 10) // 10
            h = windows.hamming(r * 2 + 1)
            f = h / h.sum()
            v = np.convolve(v, f, mode="same")

        # mask for non-nan times
        mask = ~np.isnan(self.times)

        # center time
        self.center = np.median(self.times[mask])

        # linear trace interpolation
        self.trace = interp1d(
            x=self.times[mask] - self.center,
            y=v[mask],
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

    def __call__(self, start, end):
        n = round((end - start) / self.target_period) + 1
        s = start - self.center
        y = self.trace(s + np.arange(n) * self.target_period)

        if np.isnan(y).any():
            return
        else:
            return y.astype(np.float32)
