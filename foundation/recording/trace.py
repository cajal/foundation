import numpy as np
import pandas as pd
import datajoint as dj
from djutils import link, group, merge, row_property, skip_missing
from foundation.utility import resample
from foundation.scan import timing as scan_timing, pupil as scan_pupil
from foundation.recording import trial
from foundation.schemas.pipeline import (
    pipe_fuse,
    pipe_shared,
    pipe_stim,
    pipe_eye,
    pipe_tread,
    resolve_pipe,
)
from foundation.schemas import recording as schema


# -------------- Trace --------------

# -- Trace Base --


class TraceBase:
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


# -- Trace Types --


class ScanBase(TraceBase):
    """Scan Traces"""

    @row_property
    def trials(self):
        trials = pipe_stim.Trial.proj() & self
        trials = merge(trials, trial.TrialLink.ScanTrial)
        return trial.TrialSet.get(trials)


@schema
class ScanResponse(ScanBase, dj.Lookup):
    definition = """
    -> scan_timing.Timing
    -> pipe_fuse.ScanSet.Unit
    -> pipe_shared.SpikeMethod
    """

    @row_property
    def pipe(self):
        key = dj.U("animal_id", "session", "scan_idx") & self
        return resolve_pipe(**key.fetch1())

    @row_property
    def times(self):
        times = (scan_timing.Timing & self).fetch1("scan_times")
        delay = (self.pipe.ScanSet.UnitInfo & self).fetch1("ms_delay") / 1000
        return times + delay

    @row_property
    def values(self):
        return (self.pipe.Activity.Trace & self).fetch1("trace").clip(0)


@schema
class ScanPupil(ScanBase, dj.Lookup):
    definition = """
    -> scan_pupil.PupilTrace
    """

    @row_property
    def times(self):
        return (scan_timing.Timing & self).fetch1("eye_times")

    @row_property
    def values(self):
        return (scan_pupil.PupilTrace & self).fetch1("pupil_trace")


@schema
class ScanTreadmill(ScanBase, dj.Lookup):
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


# -- Trace Link --


@link(schema)
class TraceLink:
    links = [ScanResponse, ScanPupil, ScanTreadmill]
    name = "trace"
    comment = "recording trace"


@group(schema)
class TraceSet:
    keys = [TraceLink]
    name = "traces"
    comment = "set of recording traces"


# -- Computed Trace --


@schema
class TraceTrials(dj.Computed):
    definition = """
    -> TraceLink
    ---
    -> trial.TrialSet
    """

    @skip_missing
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


@schema
class TraceSamples(dj.Computed):
    definition = """
    -> TraceTrials
    -> resample.RateLink
    -> resample.OffsetLink
    -> resample.ResampleLink
    ---
    trace           : longblob      # resampled trace
    """

    @skip_missing
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
        trace = np.concatenate(samples)
        self.insert1(dict(key, trace=trace))

    @row_property
    def trials(self):
        """
        Returns
        -------
        pd.DataFrame
            index -- trial_id
            trace -- resampled trace
        """
        trials = (TraceTrials & self).trials
        trials = merge(trials, trial.TrialSamples)

        trial_id, samples = trials.fetch("trial_id", "samples", order_by="member_id")
        *split, total = np.cumsum(samples)

        trace = self.fetch1("trace")
        assert len(trace) == total

        return pd.DataFrame(
            data={"trace": np.split(trace, split)},
            index=pd.Index(trial_id, name="trial_id"),
        )
