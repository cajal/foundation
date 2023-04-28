import numpy as np
import datajoint as dj
from djutils import link, group, merge, row_property, skip_missing
from foundation.bridge.pipeline import pipe_stim, pipe_meso, pipe_eye, pipe_tread
from foundation.recording import trial, resample

schema = dj.schema("foundation_recording")


# -------------- Trace --------------

# -- Trace Base --


class TraceBase:
    """Recording Trace"""

    @row_property
    def times(self):
        """
        Returns
        -------
        times : 1D array
            trace times
        """
        raise NotImplementedError()

    @row_property
    def values(self):
        """
        Returns
        -------
        values : 1D array
            trace values
        """
        raise NotImplementedError()

    @row_property
    def trials(self):
        """
        Returns
        -------
        trial.TrialSet
            tuple from trial.TrialSet
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
    def times(self):
        """
        Returns
        -------
        times : 1D array
            trace times
        """
        from foundation.recording.scan import scan_times

        times = scan_times(**self.scan_key)[0]
        delay = (pipe_meso.ScanSet.UnitInfo & self).fetch1("ms_delay") / 1000
        return times + delay

    @row_property
    def values(self):
        """
        Returns
        -------
        values : 1D array
            trace values
        """
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
    def times(self):
        from foundation.recording.scan import eye_times

        return eye_times(**self.scan_key)

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
    def times(self):
        from foundation.recording.scan import treadmill_times

        return treadmill_times(**self.scan_key)

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
    -> resample.RateLink
    -> resample.OffsetLink
    ---
    -> trial.TrialSet
    """

    @skip_missing
    def make(self, key):
        period = (resample.RateLink & key).link.period
        offset = (resample.OffsetLink & key).link.offset

        trace_link = (TraceLink & key).link

        times = trace_link.times
        center = np.nanmedian(times)
        start = np.nanmin(times) - center
        end = np.nanmax(times) - center

        trials = trace_link.trials.members
        trials = merge(trials, trial.TrialBounds * trial.TrialSamples & key)
        trials = trials.fetch(format="frame", order_by=trials.primary_key).reset_index()

        trials["start"] = trials["start"] - center
        trials["end"] = trials["start"] + trials["samples"] * period
        trials = trials[(trials["start"] >= start) & (trials["end"] <= end)]

        trials = trials[["trial_id"]].to_dict(orient="records")
        trials = trial.TrialSet.fill(trials)

        self.insert1(dict(**key, **trials))


# @schema
# class TraceNans(dj.Computed):
#     definition = """
#     -> TraceLink
#     -> trial.TrialSamples
#     -> resample.OffsetLink
#     ---
#     nans = NULL     : int       # number of NaNs
#     """

#     @property
#     def key_source(self):
#         return TraceLink.proj() * resample.OffsetLink.proj() & [
#             TraceLink.ScanPupil * ScanPupilType & {"pupil_type": "radius"},
#             TraceLink.ScanTreadmill,
#         ]

#     @skip_missing
#     def make(self, key):
#         from foundation.utils.trace import Nans

#         period = (resample.RateLink & key).link.period
#         offset = (resample.OffsetLink & key).link.offset
#         trace_link = (TraceLink & key).link

#         nans = Nans(trace_link.times, trace_link.values, period)
#         trials = merge(trace_link.trials, trial.TrialBounds, trial.TrialSamples)
#         trials = trials.fetch(dj.key, "start", "samples", order_by=trials.primary_key)
#         keys = []

#         for trial_key, start, samples in zip(*trials):

#             _nans = nans(start + offset, samples)

#             if _nans is None:
#                 _key = dict(nans=None, **key, **trial_key)
#             else:
#                 _key = dict(nans=_nans.sum(), **key, **trial_key)

#             keys.append(_key)

#         self.insert(keys)


# @schema
# class TraceSamples(dj.Computed):
#     definition = """
#     -> TraceLink
#     -> trial.TrialSamples
#     -> resample.OffsetLink
#     -> resample.ResampleLink
#     ---
#     trace = NULL        : longblob      # resampled trace
#     """

#     @property
#     def key_source(self):
#         return TraceLink.proj() * resample.OffsetLink.proj() * resample.ResampleLink.proj()

#     @skip_missing
#     def make(self, key):
#         period = (resample.RateLink & key).link.period
#         offset = (resample.OffsetLink & key).link.offset
#         resampler = (resample.ResampleLink & key).link.resampler
#         trace_link = (TraceLink & key).link

#         trace = resampler(trace_link.times, trace_link.values, period)
#         trials = merge(trace_link.trials, trial.TrialBounds, trial.TrialSamples)
#         trials = trials.fetch(dj.key, "start", "samples", order_by=trials.primary_key)
#         keys = []

#         for trial_key, start, samples in zip(*trials):

#             _trace = trace(start + offset, samples)

#             if _trace is None:
#                 _key = dict(trace=None, **key, **trial_key)
#             else:
#                 _key = dict(trace=_trace, **key, **trial_key)

#             keys.append(_key)

#         self.insert(keys)
