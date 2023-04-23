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

        if reduce == "max":
            self.reduce = np.max
        else:
            raise NotImplementedError(f"Reduce type '{reduce}' is not implemented.")

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
        float
            maximum gap in time between nans in trace times or values
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
