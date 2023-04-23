import numpy as np
import datajoint as dj
from djutils import link, row_property, MissingError, RestrictionError
from foundation.utils.logging import logger
from foundation.bridge.pipeline import pipe_meso, pipe_eye, pipe_tread
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
    def trial_flips(self):
        """
        Returns
        -------
        trials.TrialFlips
            tuples from trials.TrialFlips
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
    def trial_flips(self):
        from foundation.bridge.pipeline import pipe_stim

        scan_trials = pipe_stim.Trial.proj() & self
        keys = trial.TrialFlips.proj() * trial.TrialLink.ScanTrial * scan_trials

        if scan_trials - keys:
            raise MissingError("Missing trials.")

        if keys - scan_trials:
            raise RestrictionError("Unexpected trials.")

        return trial.TrialFlips & keys


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


# -- Computed Trace --


@schema
class TraceGap(dj.Computed):
    definition = """
    -> TraceLink
    -> trial.TrialLink
    -> resample.OffsetLink
    ---
    gap = NULL      : float     # nan time gap
    """

    @property
    def key_source(self):
        return TraceLink.proj() * resample.OffsetLink.proj() & [
            TraceLink.ScanPupil * ScanPupilType & {"pupil_type": "radius"},
            TraceLink.ScanTreadmill,
        ]

    def make(self, key):
        from foundation.utils.trace import Gap

        try:
            trace_link = (TraceLink & key).link
            trial_flips = trace_link.trial_flips
            gap = Gap(trace_link.times, trace_link.values)

            offset_link = (resample.OffsetLink & key).link
            offset = offset_link.offset

        except MissingError:
            logger.warn(f"Missing data. Not populating {key}")
            return

        trials = trial_flips.fetch(dj.key, "flip_start", "flip_end", order_by=trial_flips.primary_key)
        keys = []
        for trial_key, start, end in zip(*trials):

            k = dict(gap=gap(start + offset, end + offset), **key, **trial_key)
            keys.append(k)

        self.insert(keys)


@schema
class ResampledTrace(dj.Computed):
    definition = """
    -> TraceLink
    -> trial.TrialLink
    -> resample.RateLink
    -> resample.OffsetLink
    -> resample.ResampleLink
    ---
    trace = NULL        : longblob      # resampled trace
    """

    @property
    def key_source(self):
        return TraceLink.proj() * resample.RateLink.proj() * resample.OffsetLink.proj() * resample.ResampleLink.proj()

    def make(self, key):
        try:
            trace_link = (TraceLink & key).link
            trial_flips = trace_link.trial_flips
            times = trace_link.times
            values = trace_link.values

            rate_link = (resample.RateLink & key).link
            target_period = rate_link.period

            offset_link = (resample.OffsetLink & key).link
            offset = offset_link.offset

            resamp_link = (resample.ResampleLink & key).link
            resampler = resamp_link.resampler(times, values, target_period)

        except MissingError:
            logger.warn(f"Missing data. Not populating {key}")
            return

        trials = trial_flips.fetch(dj.key, "flip_start", "flip_end", order_by=trial_flips.primary_key)
        keys = []
        for trial_key, start, end in zip(*trials):

            k = dict(trace=resampler(start + offset, end + offset), **key, **trial_key)
            keys.append(k)

        self.insert(keys)
