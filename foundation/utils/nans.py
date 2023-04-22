import numpy as np
from scipy.interpolate import interp1d
from .trace import fill_nans, monotonic
from .logging import logger


class Nans:
    def __call__(self, start, end):
        """
        Parameters
        ----------
        start : float
            start time of sample
        end : float
            end time of sample

        Returns
        -------
        int
            Nans detected between start and end
        """
        raise NotImplementedError()


class DurationNans(Nans):
    """Detects duration of nans"""

    def __init__(self, times, values, reduce="max"):
        """
        Parameters
        -------
        times : 1D array
            trace times
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
        self.nans = np.zeros_like(self.ctimes)
        self.nans[idx_nan] = right(idx_nan) - left(idx_nan)

    def __call__(self, start, end):
        bounds = [
            start - self.center,
            end - self.center,
        ]
        bounds = self.index(bounds)

        if np.isnan(bounds).any():
            return
        else:
            i, j = map(int, bounds)
            nans = self.nans[i : j + 1]
            return self.reduce(nans)
