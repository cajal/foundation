import numpy as np
import datajoint as dj
from djutils import link, group, merge, row_property, skip_missing
from foundation.bridge.pipeline import pipe_stim, pipe_meso, pipe_eye, pipe_tread
from foundation.recording import trial, resample

from time import time

schema = dj.schema("foundation_recording")


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
    def bounds(self):
        """
        Returns
        -------
        float
            start time of trace
        end
            end time of trace
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
    """Scan Trials"""

    @row_property
    def scan_key(self):
        key = ["animal_id", "session", "scan_idx"]
        return dict(zip(key, self.fetch1(*key)))

    @row_property
    def trials(self):
        trials = pipe_stim.Trial.proj() & self
        trials = merge(trials, trial.TrialLink.ScanTrial)
        return trial.TrialSet.get(trials)


@schema
class MesoActivity(ScanBase, dj.Lookup):
    definition = """
    -> pipe_meso.Activity.Trace
    """

    @row_property
    def bounds(self):
        from foundation.recording.scan import ScanTimes

        return merge(self, ScanTimes).fetch1("start", "end")

    @row_property
    def times(self):
        from foundation.recording.scan import ScanTimes

        times = merge(self, ScanTimes).fetch1("times")
        delay = (pipe_meso.ScanSet.UnitInfo & self).fetch1("ms_delay") / 1000
        return times + delay

    @row_property
    def values(self):
        return (pipe_meso.Activity.Trace & self).fetch1("trace").clip(0)


@schema
class ScanPupilType(dj.Lookup):
    definition = """
    pupil_type      : varchar(64)   # fitted scan pupil type
    """


@schema
class ScanPupil(ScanBase, dj.Lookup):
    definition = """
    -> pipe_eye.FittedPupil
    -> ScanPupilType
    """

    @row_property
    def bounds(self):
        from foundation.recording.scan import EyeTimes

        return merge(self, EyeTimes).fetch1("start", "end")

    @row_property
    def times(self):
        from foundation.recording.scan import EyeTimes

        return merge(self, EyeTimes).fetch1("times")

    @row_property
    def values(self):
        fits = pipe_eye.FittedPupil.Circle & self
        pupil_type = self.fetch1("pupil_type")

        if pupil_type == "radius":
            return fits.fetch("radius", order_by="frame_id")

        elif pupil_type in ["center_x", "center_y"]:

            center = fits.fetch("center", order_by="frame_id")
            if pupil_type == "center_x":
                return np.array([np.nan if c is None else c[0] for c in center])
            else:
                return np.array([np.nan if c is None else c[1] for c in center])

        else:
            raise NotImplementedError(f"Pupil type '{pupil_type}' not implemented.")


@schema
class ScanTreadmill(ScanBase, dj.Lookup):
    definition = """
    -> pipe_tread.Treadmill
    """

    @row_property
    def bounds(self):
        from foundation.recording.scan import TreadmillTimes

        return merge(self, TreadmillTimes).fetch1("start", "end")

    @row_property
    def times(self):
        from foundation.recording.scan import TreadmillTimes

        return merge(self, TreadmillTimes).fetch1("times")

    @row_property
    def values(self):
        return (pipe_tread.Treadmill & self).fetch1("treadmill_vel")


# -- Trace Link --


@link(schema)
class TraceLink:
    links = [MesoActivity, ScanPupil, ScanTreadmill]
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


@schema
class TraceBounds(dj.Computed):
    definition = """
    -> TraceTrials
    -> resample.RateLink
    -> resample.OffsetLink
    ---
    -> trial.TrialSet
    """

    @skip_missing
    def make(self, key):
        period = (resample.RateLink & key).link.period
        offset = (resample.OffsetLink & key).link.offset

        tmin, tmax = (TraceLink & key).link.bounds
        center = (tmin + tmax) / 2
        tmin -= center
        tmax -= center

        trials = trial.TrialSet & (TraceTrials & key)
        trials = merge(trials.members, trial.TrialBounds, trial.TrialSamples & key)

        trial_id, start, end, samples = trials.fetch("trial_id", "start", "end", "samples")
        start = start - center + offset
        end = start + samples * period

        oob = (start < tmin) | (end > tmax)
        oob = [dict(trial_id=tid) for tid in trial_id[oob]]
        oob = trial.TrialSet.fill(oob, prompt=False, silent=True)

        key["trials_id"] = oob["trials_id"]
        self.insert1(key)

    @row_property
    def trials(self):
        include = (trial.TrialSet & TraceTrials & self.proj()).members
        exclude = (trial.TrialSet & self).members

        return (trial.TrialLink & include).proj() - (trial.TrialLink & exclude).proj()


# @schema
# class TraceSamples(dj.Computed):
#     definition = """
#     -> TraceBounds
#     -> resample.ResampleLink
#     ---
#     -> trial.TrialSet
#     trace               : longblob      # resampled trace
#     """

#     @skip_missing
#     def make(self, key):
#         trials = (TraceBounds & key).trials

#         trial_samples = merge(trials, trial.TrialBounds, trial.TrialSamples & key)
#         start, samples = trial_samples.fetch("start", "samples", order_by="trial_id")

#         trials_key = trial.TrialSet.fill(trials, prompt=False, silent=True)

#         period = (resample.RateLink & key).link.period
#         offset = (resample.OffsetLink & key).link.offset

#         trace_link = (TraceLink & key).link
#         resampler = (resample.ResampleLink & key).link.resampler(
#             trace_link.times,
#             trace_link.values,
#             period,
#         )
#         trace = np.concatenate([resampler(t + offset, n) for t, n in zip(start, samples)])

#         self.insert1(dict(trace=trace, **key, **trials_key))


