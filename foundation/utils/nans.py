import numpy as np
from scipy.interpolate import interp1d
from .traces import fill_nans, TraceTimes
from .logging import logger


class NanDetector(TraceTimes):
    def __call__(self, start, end):
        """
        Parameters
        ----------
        start : float
            start time on initialized clock
        end : float
            end time on initialized clock

        Returns
        -------
        int
            Nans detected between start and end
        """
        raise NotImplementedError()


class ConsecutiveNans(NanDetector):
    def init(self):
        # nans in either times or trace
        times_nan = np.isnan(self.times)
        trace_nan = np.isnan(self.trace)
        nan = times_nan | trace_nan

        # nan ranges
        pad = np.concatenate([[0], nan, [0]])
        delt = np.nonzero(np.diff(pad))[0]

        # consecutive nan trace
        self.nans = np.zeros_like(self.trace, dtype=int)
        for a, b in delt.reshape(-1, 2):
            self.nans[a:b] = b - a

        # fill nans in times
        if times_nan.any():
            logger.info("Found nans in times. Filling with linear interpolation.")
            self.times = fill_nans(self.times)

    def __call__(self, start, end):
        i = np.searchsorted(t, x, side="right") - 1
        j = np.searchsorted(t, y, side="left") + 1
        return self.nans[i:j].max()
