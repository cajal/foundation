import numpy as np
from scipy.interpolate import interp1d
from .traces import Samples
from .logging import logger


class Nans(Samples):
    def __call__(self, start, end):
        """
        Parameters
        ----------
        start : float
            start time on initial clock
        end : float
            end time on initial clock

        Returns
        -------
        int
            Nans detected between start and end
        """
        raise NotImplementedError()


class ConsecutiveNans(Nans):
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

        # nearest index
        self.index = interp1d(self.times, np.arange(len(self.times)), kind="nearest")

    def __call__(self, start, end):
        i, j = map(int, map(self.index, map(self.clock, [start, end])))
        return self.nans[i:j].max()
