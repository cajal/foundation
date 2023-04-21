import numpy as np
from scipy.interpolate import interp1d
from .errors import OutOfBounds
from .logging import logger


def truncate(*arrays, tolerance=1):
    """Truncates arrays to the same length

    Parameters
    ----------
    traces : Tuple[Trace]
        traces that possibly differ in length
    tolerance : int
        tolerance for length mismatches

    Returns
    -------
    Tuple[Trace]
        traces of the same length
    """
    lengths = tuple(map(len, arrays))
    min_len = min(lengths)
    max_len = max(lengths)

    if max_len - min_len > tolerance:
        raise ValueError(f"Traces differ in length by more than {tolerance}")

    if max_len > min_len:
        logger.info(f"Truncating {max_len - min_len} frames")

    return tuple(trace[:min_len] for trace in arrays)


class Trace:
    """1D trace"""

    def __init__(self, array, nan=False, monotonic=False, copy=True):
        """
        Parameters
        ----------
        array : array-like
            1D array
        nan : bool
            whether nans are allowed in the array
        monotonic : bool
            whether values monotonically increase
        copy : bool
            whether array data is copied
        """
        self.array = np.array(array, copy=copy)
        self.nan = bool(nan)
        self.monotonic = bool(monotonic)

        if self.array.ndim != 1:
            raise ValueError("Trace must be 1D.")

        if not self.nan and np.isnan(self.array).any():
            raise ValueError("Nans found in trace.")

        if self.monotonic:
            t = self.array[~self.nan_mask] if self.nan else self.array
            if np.diff(t).min() < 0:
                raise ValueError("Values do not monotonically increase.")

    @property
    def nan_mask(self):
        """
        Returns
        -------
        1D array (dtype = bool) | None
            boolean mask indicating nans
        """
        if not self.nan:
            return

        _nan_mask = getattr(self, "_nan_mask", None)
        if _nan_mask is None:
            _nan_mask = self._nan_mask = np.isnan(self.array)

        return _nan_mask

    @property
    def filled(self):
        """
        Returns
        -------
        Array
            Array with all nans filled
        """
        if self.nan:
            if self.nan_mask.all():
                raise ValueError("All values are nan.")

            if self.nan_mask.any():
                logger.info("Nans in trace. Filling with linear interpolation.")

                new = self.array.copy()
                new[self.nan_mask] = np.interp(
                    x=np.nonzero(self.nan_mask)[0],
                    xp=np.nonzero(~self.nan_mask)[0],
                    fp=self.array[~self.nan_mask],
                )
                return Trace(array=new, nan=False, copy=False, monotonic=self.monotonic)

        logger.info("No nans in trace.")
        return self

    @property
    def median(self):
        ret = getattr(self, "_median", None)

        if ret is None:
            ret = self._median = np.nanmedian(self.array)

        return ret

    @property
    def min(self):
        ret = getattr(self, "_min", None)

        if ret is None:
            ret = self._min = np.nanmin(self.array)

        return ret

    @property
    def max(self):
        ret = getattr(self, "_max", None)

        if ret is None:
            ret = self._max = np.nanmax(self.array)

        return ret

    def __len__(self):
        return len(self.array)

    def __getitem__(self, key):
        ret = self.array[key]

        if ret.ndim > 0:
            ret = Trace(self.array[key], nan=self.nan, copy=True, monotonic=self.monotonic)

        return ret


class Resample:
    def __init__(self, times, values):
        """
        Parameters
        ----------
        times : Array
            trace times
        values : Array
            trace values
        """
        if not times.monotonic:
            raise ValueError("Times are not monotonic.")

        # trace values
        self.values = values

        # initial clock
        self.times = times

        # centered clock
        self.centered_times = times.array - times.median

    def clock(self, times):
        """
        Parameters
        ----------
        times : array-like
            times on initial clock

        array-like
            times on centered clock
        """
        if np.nanmin(times) < self.times.min:
            raise OutOfBounds("Provided time is less than registered time values.")

        if np.nanmax(times) > self.times.max:
            raise OutOfBounds("Provided time is greater than registered time values.")

        return times - self.times.median

    def __call__(self, times):
        """
        Parameters
        ----------
        times : array-like
            times on initial clock

        Returns
        -------
        1D array
            resampled values, same length as times
        """
        raise NotImplementedError()


# class Sample:
#     def __init__(self, time, traces, target_period, offsets=None, tolerance=0, kind="hamming", **kwargs):
#         """
#         Parameters
#         ----------
#         time : 1D array
#             time values
#         traces : Sequence[1D array]
#             trace values
#         target_period : float
#             target sampling period
#         offsets : Sequence[float] | None
#             time offsets for each trace
#         tolerace : int
#             tolerance for time and trace length mismatches
#         kind : str
#             specifies the kind of sampling
#         kwargs : dict
#             additional sampling options
#         """
#         # truncate time and trace arrays to be the same length
#         self.time, *traces = truncate(time, *traces, tolerance=tolerance)

#         # verify time
#         assert np.all(np.isfinite(self.time)), "Non-finite values found in time"
#         assert np.all(np.diff(self.time) >= 0), "Time is not monotonically increasing"

#         # target sampling period
#         self.target_period = float(target_period)

#         # timing offset
#         if offsets is None:
#             self.offsets = [0] * len(traces)
#         else:
#             self.offsets = list(map(float, offsets))
#             assert len(self.offsets) == len(traces), "Unequal numbers of traces and offsets"

#         # center for numerical stability
#         self.t0 = np.median(self.time)
#         self.shift = lambda time: time - self.t0

#         # consecutive nans in original trace
#         interp = lambda trace, offset: interp1d(
#             x=self.shift(self.time + offset),
#             y=consecutive_nans(trace),
#             kind="nearest",
#             fill_value=np.nan,
#             bounds_error=False,
#             assume_sorted=True,
#             copy=False,
#         )
#         self.nans = list(starmap(interp, zip(traces, self.offsets)))

#         # sampling kind
#         self.kind = str(kind)

#         # linear interpolation with hamming filtering
#         if self.kind == "hamming":

#             # fill nans
#             traces = map(fill_nans, traces)

#             # low-pass if source period < target period
#             source_period = np.median(np.diff(self.time))
#             if source_period < self.target_period:
#                 logger.info("Low-pass filtering traces with hamming filter")

#                 filt = lowpass_filter(source_period, self.target_period, "hamming")
#                 convolve = lambda trace: np.convolve(trace, filt, mode="same")
#                 traces = map(convolve, traces)

#             # linear interpolation
#             interp = lambda trace, offset: interp1d(
#                 x=self.shift(self.time + offset),
#                 y=trace,
#                 kind="linear",
#                 fill_value=np.nan,
#                 bounds_error=False,
#                 assume_sorted=True,
#                 copy=False,
#             )
#             self.traces = list(starmap(interp, zip(traces, self.offsets)))

#         else:
#             raise NotImplementedError(f"Sampling kind '{kind}' not implemented")

#         # warn of unused kwargs
#         ignored = set(kwargs)
#         if ignored:
#             logger.warning(f"Ignoring kwargs {ignored}")

#     def interpolants(self, start, end):
#         """
#         Parameters
#         ----------
#         start : float
#             start time
#         end : float
#             end time

#         Returns
#         -------
#         1D array
#             interpolant values
#         """
#         return np.arange(self.shift(start), self.shift(end), self.target_period)

#     def __call__(self, start, end):
#         """Sample traces

#         Parameters
#         ----------
#         start : float
#             start time
#         end : float
#             end time

#         Returns
#         -------
#         Tuple[1D array]
#             sampled traces
#         """
#         t = self.interpolants(start, end)
#         return tuple(trace(t) for trace in self.traces)

#     def consecutive_nans(self, start, end):
#         """Consecutive NaNs in original trace

#         Parameters
#         ----------
#         start : float
#             start time
#         end : float
#             end time

#         Returns
#         -------
#         Tuple[t]
#             Maximum number of consecutive NaNs per trace
#         """
#         t = self.interpolants(start, end)
#         nans = (nan(t).max() for nan in self.nans)
#         return tuple(None if np.isnan(n) else int(n) for n in nans)
