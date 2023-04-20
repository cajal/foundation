import numpy as np
from scipy.interpolate import interp1d
from .errors import OutOfBounds
from .logging import logger


class Trace:
    """1D trace"""

    def __init__(self, array, nans=False, copy=True):
        """
        Parameters
        ----------
        array : array-like
            1D trace array
        nans : bool
            whether nans are allowed in the trace
        copy : bool
            whether trace data is copied
        """
        self.array = np.array(array, copy=copy)
        self.nans = bool(nans)

        if self.array.ndim != 1:
            raise ValueError("Trace must be 1D.")

        if not self.nans and np.isnan(self.array).any():
            raise ValueError("Nans found in trace.")

    @property
    def nan_mask(self):
        """
        Returns
        -------
        1D array (dtype = bool) | None
            boolean mask indicating nans
        """
        if not self.nans:
            return

        _nan_mask = getattr(self, "_nan_mask", None)
        if _nan_mask is None:
            _nan_mask = self._nan_mask = np.isnan(self.array)

        return _nan_mask

    @property
    def median(self):
        """
        Returns
        -------
        float-like
            median of the trace
        """
        _median = getattr(self, "_median", None)
        if _median is None:
            _median = self._median = np.nanmedian(self.array)

        return _median

    @property
    def centered(self):
        """
        Returns
        -------
        1D array
            centered trace array
        """
        return self.center(self.array)

    def center(self, trace):
        """
        Parameters
        ----------
        1D array
            trace array

        Returns
        -------
        1D array
            centers the provided trace array
        """
        return trace - self.median

    def uncenter(self, trace):
        """
        Parameters
        ----------
        1D array
            trace array

        Returns
        -------
        1D array
            uncenters the provided trace array
        """
        return trace + self.median

    def __len__(self):
        return len(self.array)

    def __getitem__(self, key):
        ret = self.array[key]

        if ret.ndim > 0:
            ret = self.__class__(self.array[key], nans=self.nans, copy=True)

        return ret


class Times(Trace):
    def __init__(self, times, nans=False, copy=True):
        super().__init__(array=times, nans=nans, copy=copy)

        t = self.array[~self.nan_mask] if self.nans else self.array
        incr = np.diff(t) > 0
        if not incr.all():
            raise ValueError("Values do not monotonically increase.")


def fill_nans(trace, inplace=False): # TODO
    """
    Parameters
    ----------
    trace : 1D array
        trace values
    inplace : bool
        inplace modification of trace

    Returns
    -------
    1D array
        trace with nan values interpolated
    """
    if not inplace:
        trace = trace.copy()

    nan = np.isnan(trace)
    trace[nan] = 0 if nan.all() else np.interp(np.nonzero(nan)[0], np.nonzero(~nan)[0], trace[~nan])
    return trace


def truncate(*traces, tolerance=1):
    """Truncates traces to the same length

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
    lengths = tuple(map(len, traces))
    min_len = min(lengths)
    max_len = max(lengths)

    if max_len - min_len > tolerance:
        raise ValueError(f"Traces differ in length by more than {tolerance}")

    if max_len > min_len:
        logger.info(f"Truncating {max_len - min_len} frames")

    return tuple(trace[:min_len] for trace in traces)


class TraceTimes:
    def __init__(self, times, trace):
        """
        Parameters
        ----------
        times : 1D array
            time of each point in trace, monotonically increasing
        trace : 1D array
            trace values, same length as times
        """
        if len(times) != len(trace):
            raise ValueError("times and trace arrays must be the same length")

        if not np.nanmin(np.diff(times)) > 0:
            raise ValueError("times must be monotically increasing")

        self.t0 = np.nanmedian(times)
        self.tmin = np.nanmin(times)
        self.tmax = np.nanmax(times)
        self.times = self.clock(times, verify=False)
        self.trace = trace
        self.init()

    def init(self):
        pass

    def clock(self, time, verify=True):
        """
        Parameter
        ---------
        time : 1D array
            time on initialized clock
        verify : bool
            verify time is within bounds

        Returns
        -------
            time on internal clock (centered by median for numerical stability)
        """
        if verify:
            if np.min(time) < self.tmin or np.max(time) > self.tmax:
                raise OutOfBounds("Time is out of bounds.")

        return time - self.t0


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
