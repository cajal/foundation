import numpy as np
import pandas as pd
from djutils import merge, row_property, row_method
from operator import add
from functools import reduce
from foundation.recording import trial
from foundation.utility import resample
from foundation.scan import timing as scan_timing, pupil as scan_pupil
from foundation.schemas.pipeline import (
    pipe_fuse,
    pipe_shared,
    pipe_stim,
    pipe_tread,
    resolve_pipe,
)
from foundation.schemas import recording as schema


# -------------- Trace --------------

# -- Trace Base --


class _Trace:
    """Recording Trace"""

    @row_property
    def trials(self):
        """
        Returns
        -------
        trial.TrialSet
            tuple from trial.TrialSet
        """
        raise NotImplementedError()

    @row_property
    def times(self):
        """
        Returns
        -------
        1D array
            trace times
        """
        raise NotImplementedError()

    @row_property
    def values(self):
        """
        Returns
        -------
        1D array
            trace values
        """
        raise NotImplementedError()

    @row_property
    def homogeneous(self):
        """
        Returns
        -------
        bool
            homogeneous transformation of trace values
        """
        raise NotImplementedError()


# -- Trace Types --


class _Scan(_Trace):
    """Scan Trace"""

    @row_property
    def trials(self):
        trials = pipe_stim.Trial.proj() & self
        trials = merge(trials, trial.TrialLink.ScanTrial)
        trials = trial.TrialSet.fill(trials, prompt=False, silent=True)
        return trial.TrialSet & trials


@schema.lookup
class ScanResponse(_Scan):
    definition = """
    -> scan_timing.Timing
    -> pipe_fuse.ScanSet.Unit
    -> pipe_shared.SpikeMethod
    """

    @row_property
    def pipe(self):
        return resolve_pipe(self)

    @row_property
    def times(self):
        times = (scan_timing.Timing & self).fetch1("scan_times")
        delay = (self.pipe.ScanSet.UnitInfo & self).fetch1("ms_delay") / 1000
        return times + delay

    @row_property
    def values(self):
        return (self.pipe.Activity.Trace & self).fetch1("trace").clip(0)

    @row_property
    def homogeneous(self):
        return True


@schema.lookup
class ScanPupil(_Scan):
    definition = """
    -> scan_pupil.PupilTrace
    """

    @row_property
    def times(self):
        return (scan_timing.Timing & self).fetch1("eye_times")

    @row_property
    def values(self):
        return (scan_pupil.PupilTrace & self).fetch1("pupil_trace")

    @row_property
    def homogeneous(self):
        return False


@schema.lookup
class ScanTreadmill(_Scan):
    definition = """
    -> scan_timing.Timing
    -> pipe_tread.Treadmill
    """

    @row_property
    def times(self):
        return (scan_timing.Timing & self).fetch1("treadmill_times")

    @row_property
    def values(self):
        return (pipe_tread.Treadmill & self).fetch1("treadmill_vel")

    @row_property
    def homogeneous(self):
        return True


# -- Trace Link --


@schema.link
class TraceLink:
    links = [ScanResponse, ScanPupil, ScanTreadmill]
    name = "trace"
    comment = "recording trace"


@schema.set
class TraceSet:
    keys = [TraceLink]
    name = "traces"
    comment = "set of recording traces"


# -- Computed Trace --


@schema.computed
class TraceHomogeneous:
    definition = """
    -> TraceLink
    ---
    homogeneous     : bool      # homogeneous tranformation
    """

    def make(self, key):
        key["homogeneous"] = (TraceLink & key).link.homogeneous
        self.insert1(key)


@schema.computed
class TraceTrials:
    definition = """
    -> TraceLink
    ---
    -> trial.TrialSet
    """

    def make(self, key):
        key["trials_id"] = (TraceLink & key).link.trials.fetch1("trials_id")
        self.insert1(key)

    @row_property
    def trials(self):
        """
        Returns
        -------
        trial.TrialSet.Member
            members from trial set
        """
        return (trial.TrialSet & self).members


@schema.computed
class TraceSamples:
    definition = """
    -> TraceTrials
    -> resample.RateLink
    -> resample.OffsetLink
    -> resample.ResampleLink
    ---
    trace           : longblob          # resampled trace
    samples         : int unsigned      # number of samples
    nans            : int unsigned      # number of nans
    """

    def make(self, key):
        # resampling
        period = (resample.RateLink & key).link.period
        offset = (resample.OffsetLink & key).link.offset
        resampler = (resample.ResampleLink & key).link.resampler

        # trace resampler
        trace_link = (TraceLink & key).link
        r = resampler(times=trace_link.times, values=trace_link.values, target_period=period)

        # trials
        trials = (TraceTrials & key).trials
        trials = merge(trials, trial.TrialBounds)

        # sample trials, ordered by member_id
        start, end = trials.fetch("start", "end", order_by="member_id")
        samples = [r(s, e, offset) for s, e in zip(start, end)]

        # concatenate samples
        trace = np.concatenate(samples).astype(np.float32)
        samples = len(trace)
        nans = np.isnan(trace).sum()

        # insert key
        self.insert1(dict(key, trace=trace, samples=samples, nans=nans))

    @row_property
    def trials(self):
        """
        Returns
        -------
        pd.DataFrame
            index -- trial_id
            trace -- resampled trace
        """
        # fetch data
        key, trace, samples = self.fetch1("KEY", "trace", "samples")

        # trials
        trials = merge((TraceTrials & key).trials, trial.TrialSamples & key)

        # trial samples, ordered by member_id
        trial_id, trial_samples = trials.fetch("trial_id", "samples", order_by="member_id")

        # trace split indices
        *split, total = np.cumsum(trial_samples)
        assert total == samples == len(trace)

        # dataframe containing trial samples
        return pd.DataFrame(
            data={"trace": np.split(trace, split)},
            index=pd.Index(trial_id, name="trial_id"),
        )


# -------------- Trace Filter --------------


@schema.filter_link
class TraceFilterLink:
    filters = []
    name = "trace_filter"
    comment = "recording trace filter"


@schema.filter_link_set
class TraceFilterSet:
    filter_link = TraceFilterLink
    name = "trace_filters"
    comment = "set of recording trace filters"
