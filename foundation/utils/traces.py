import numpy as np
from scipy.interpolate import interp1d
from itertools import starmap
from .signal import lowpass_filter
from .logging import logger


def consecutive_nans(trace):
    """
    Parameters
    ----------
    trace : 1D array
        trace values

    Returns
    -------
    1D array
        consecutive NaNs
    """
    nan = np.isnan(trace)
    pad = np.concatenate([[0], nan, [0]])
    delt = np.nonzero(np.diff(pad))[0]
    nans = np.zeros_like(trace, dtype=int)

    for a, b in delt.reshape(-1, 2):
        nans[a:b] = b - a

    return nans


def fill_nans(trace, inplace=False):
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
    traces : Tuple[1D array]
        traces that possibly differ in length
    tolerance : int
        tolerance for length mismatches

    Returns
    -------
    Tuple[1D array]
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


class Sample:
    def __init__(self, time, traces, target_period, offsets=None, tolerance=0, kind="hamming", **kwargs):
        """
        Parameters
        ----------
        time : 1D array
            time values
        traces : Sequence[1D array]
            trace values
        target_period : float
            target sampling period
        offsets : Sequence[float] | None
            time offsets for each trace
        tolerace : int
            tolerance for time and trace length mismatches
        kind : str
            specifies the kind of sampling
        kwargs : dict
            additional sampling options
        """
        # truncate time and trace arrays to be the same length
        self.time, *traces = truncate(time, *traces, tolerance=tolerance)

        # verify time
        assert np.all(np.isfinite(self.time)), "Non-finite values found in time"
        assert np.all(np.diff(self.time) >= 0), "Time is not monotonically increasing"

        # target sampling period
        self.target_period = float(target_period)

        # timing offset
        if offsets is None:
            self.offsets = [0] * len(traces)
        else:
            self.offsets = list(map(float, offsets))
            assert len(self.offsets) == len(traces), "Unequal numbers of traces and offsets"

        # center for numerical stability
        self.t0 = np.median(self.time)
        self.shift = lambda time: time - self.t0

        # consecutive nans in original trace
        interp = lambda trace, offset: interp1d(
            x=self.shift(self.time + offset),
            y=consecutive_nans(trace),
            kind="nearest",
            fill_value=np.nan,
            bounds_error=False,
            assume_sorted=True,
            copy=False,
        )
        self.nans = list(starmap(interp, zip(traces, self.offsets)))

        # sampling kind
        self.kind = str(kind)

        # linear interpolation with hamming filtering
        if self.kind == "hamming":

            # fill nans
            traces = map(fill_nans, traces)

            # low-pass if source period < target period
            source_period = np.median(np.diff(self.time))
            if source_period < self.target_period:
                logger.info("Low-pass filtering traces with hamming filter")

                filt = lowpass_filter(source_period, self.target_period, "hamming")
                convolve = lambda trace: np.convolve(trace, filt, mode="same")
                traces = map(convolve, traces)

            # linear interpolation
            interp = lambda trace, offset: interp1d(
                x=self.shift(self.time + offset),
                y=trace,
                kind="linear",
                fill_value=np.nan,
                bounds_error=False,
                assume_sorted=True,
                copy=False,
            )
            self.traces = list(starmap(interp, zip(traces, self.offsets)))

        else:
            raise NotImplementedError(f"Sampling kind '{kind}' not implemented")

        # warn of unused kwargs
        ignored = set(kwargs)
        if ignored:
            logger.warning(f"Ignoring kwargs {ignored}")

    def interpolants(self, start, end):
        """
        Parameters
        ----------
        start : float
            start time
        end : float
            end time

        Returns
        -------
        1D array
            interpolant values
        """
        return np.arange(self.shift(start), self.shift(end), self.target_period)

    def __call__(self, start, end):
        """Sample traces

        Parameters
        ----------
        start : float
            start time
        end : float
            end time

        Returns
        -------
        Tuple[1D array]
            sampled traces
        """
        t = self.interpolants(start, end)
        return tuple(trace(t) for trace in self.traces)

    def consecutive_nans(self, start, end):
        """Consecutive NaNs in original trace

        Parameters
        ----------
        start : float
            start time
        end : float
            end time

        Returns
        -------
        Tuple[t]
            Maximum number of consecutive NaNs per trace
        """
        t = self.interpolants(start, end)
        nans = (nan(t).max() for nan in self.nans)
        return tuple(None if np.isnan(n) else int(n) for n in nans)
