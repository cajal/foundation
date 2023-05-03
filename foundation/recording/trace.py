import pandas as pd
from djutils import merge, row_property, row_method, RestrictionError
from foundation.recording import trial
from foundation.scan import experiment as scan_exp, pupil as scan_pupil
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
    def trial_set(self):
        """
        Returns
        -------
        foundation.recording.trial.TrialSet
            tuple
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
    def trial_set(self):
        key = pipe_stim.Trial.proj() & self
        key = merge(key, trial.TrialLink.ScanTrial)
        key = trial.TrialSet.fill(key, prompt=False, silent=True)
        return trial.TrialSet & key


@schema.lookup
class ScanUnit(_Scan):
    definition = """
    -> scan_exp.Scan
    -> pipe_fuse.ScanSet.Unit
    -> pipe_shared.SpikeMethod
    """

    @row_property
    def times(self):
        times = (scan_exp.Scan & self).fetch1("scan_times")
        delay = (resolve_pipe(self).ScanSet.UnitInfo & self).fetch1("ms_delay") / 1000
        return times + delay

    @row_property
    def values(self):
        return (resolve_pipe(self).Activity.Trace & self).fetch1("trace").clip(0)

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
        return (scan_exp.Scan & self).fetch1("eye_times")

    @row_property
    def values(self):
        return (scan_pupil.PupilTrace & self).fetch1("pupil_trace")

    @row_property
    def homogeneous(self):
        return False


@schema.lookup
class ScanTreadmill(_Scan):
    definition = """
    -> scan_exp.Scan
    -> pipe_tread.Treadmill
    """

    @row_property
    def times(self):
        return (scan_exp.Scan & self).fetch1("treadmill_times")

    @row_property
    def values(self):
        return (pipe_tread.Treadmill & self).fetch1("treadmill_vel")

    @row_property
    def homogeneous(self):
        return True


# -- Trace Link --


@schema.link
class TraceLink:
    links = [ScanUnit, ScanPupil, ScanTreadmill]
    name = "trace"
    comment = "recording trace"

    @row_method
    def resampled_trials(self, trial_links, rate_link, offset_link, resample_link):
        """
        Parameters
        ----------
        trial_links : foundation.recording.trial.TrialLink
            tuple(s)
        rate_link : foundation.utility.resample.RateLink
            tuple
        offset_link : foundation.utility.resample.OffsetLink
            tuple
        resample_link : foundation.utility.resample.ResampleLink
            tuple

        Returns
        -------
        pd.Series
            index -- trial_id
            data -- resampled trace
        """
        # ensure trials are valid
        valid_trials = trial.TrialSet & merge(self, TraceTrials)
        valid_trials = valid_trials.members

        if trial_links - valid_trials:
            raise RestrictionError("Requested trials do not belong to the trace.")

        # resampling period, offset, method
        period = rate_link.link.period
        offset = offset_link.link.offset
        resample = resample_link.link.resample

        # trace resampling function
        trace = self.link
        f = resample(times=trace.times, values=trace.values, target_period=period)

        # resampled trials
        trial_timing = merge(trial_links, trial.TrialBounds)
        trial_ids, starts, ends = trial_timing.fetch("trial_id", "start", "end", order_by="start")
        samples = [f(a, b, offset) for a, b in zip(starts, ends)]

        # pandas Series containing resampled trials
        return pd.Series(
            data=samples,
            index=pd.Index(trial_ids, name="trial_id"),
        )


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
        key["trials_id"] = (TraceLink & key).link.trial_set.fetch1("trials_id")
        self.insert1(key)


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
