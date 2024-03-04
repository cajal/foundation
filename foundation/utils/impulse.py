import numpy as np
from .resample import monotonic


# -- Impulse Interface --


class Impulse:
    """Impulse"""

    def __init__(self, times, values, target_offset=0):
        """
        Parameters
        ----------
        times : 1D array
            trace times, monotonically increasing
        values : ND array
            trace values, first dim same length as times
        target_offset : float
            target offset
        """
        if not times.ndim == 1:
            raise ValueError("Times and Values must be 1D")

        if times.size != len(values):
            raise ValueError("Times and Values are incompatible sizes")

        if not monotonic(times):
            raise ValueError("Times do not monotonically increase.")

        self.times = times
        self.values = values
        self.target_offset = target_offset

    def __call__(self, start, end):
        """
        Parameters
        ----------
        start : float
            target start time
        end : float
            target end time

        Returns
        -------
        float | array
            target value
        """
        raise NotImplementedError()


# -- Impulse Types --


class Box(Impulse):
    """Box Impulse"""

    def __call__(self, start, end):
        i = np.searchsorted(self.times, self.target_offset + start, side="left")
        j = np.searchsorted(self.times, self.target_offset + end, side="right")

        v = self.values[i:j]
        return np.mean(v, axis=0)
