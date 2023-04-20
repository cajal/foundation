import numpy as np
import datajoint as dj
from djutils import link, group, method, row_method, row_property, MissingError, RestrictionError
from foundation.utils.logging import logger
from foundation.utils.traces import truncate
from foundation.recordings import trials

pipe_meso = dj.create_virtual_module("pipe_meso", "pipeline_meso")
pipe_eye = dj.create_virtual_module("pipe_eye", "pipeline_eye")
pipe_tread = dj.create_virtual_module("pipe_tread", "pipeline_treadmill")
schema = dj.schema("foundation_recordings")


# -------------- Trace --------------

# -- Trace Base --


class TraceBase:
    """Recording Trace"""

    @row_property
    def times_trace(self):
        """
        Returns
        -------
        1D array
            recording times
        1D array
            recording trace

        IMPORTANT : arrays must be the same length
        """
        raise NotImplementedError()

    @row_property
    def trials(self):
        """
        Returns
        -------
        trials.Trial
            tuples from the trials.Trial table
        """
        raise NotImplementedError()

    @row_property
    def trial_flips(self):
        """
        Returns
        -------
        Iterator[tuple[trials.Trial, 1D array]]
            yields
                trials.Trial
                    recording trial
                1D array
                    recording times of each stimulus flip
        """
        raise NotImplementedError()


# -- Trace Types --


class ScanBase(TraceBase):
    """Scan Trials"""

    @row_property
    def trials(self):
        key = trials.TrialsLink.ScanTrials * trials.ScanTrials & self
        return (trials.Trials & key).trials


@schema
class MesoActivity(ScanBase, dj.Lookup):
    definition = """
    -> pipe_meso.Activity.Trace
    """

    @row_property
    def times_trace(self):
        from foundation.recordings.scan import stimulus_times

        # scan key
        key = ["animal_id", "session", "scan_idx"]
        key = dict(zip(key, self.fetch1(*key)))

        # times on stimulus clock
        times = stimulus_times(**key)

        # imaging delay
        delay = (pipe_meso.ScanSet.UnitInfo & self).fetch1("ms_delay") / 1000

        # activity trace
        trace = (pipe_meso.Activity.Trace & self).fetch1("trace")

        # trim to same length
        times, trace = truncate(times, trace, tolerance=1)

        return times + delay, trace

    @row_property
    def trial_flips(self):
        keys = self.trials
        for key in keys.fetch(dj.key, order_by=keys.primary_key):

            trial = trials.Trial & key
            flips = trial.flips

            yield trial, flips


class ScanBehaviorTraceBase(ScanBase):
    """Scan Behavior Trace --- stimulus time -> behavior time"""

    @row_property
    def trial_flips(self):
        from foundation.recordings.scan import stimulus_times, behavior_times
        from foundation.utils.splines import CenteredSpline

        # scan key
        key = ["animal_id", "session", "scan_idx"]
        key = dict(zip(key, self.fetch1(*key)))

        # times on stimulus and behavior clocks
        stim_times = stimulus_times(**key)
        beh_times = behavior_times(**key)

        # stimulus -> behavior time
        times = CenteredSpline(stim_times, beh_times, k=1, ext=3)

        # yield trials
        keys = self.trials
        for key in keys.fetch(dj.key, order_by=keys.primary_key):

            trial = trials.Trial & key
            flips = times(trial.flips)

            yield trial, flips


@schema
class ScanPupilType(dj.Lookup):
    definition = """
    pupil_type          : varchar(64)   # fitted scan pupil type
    pupil_attribute     : varchar(64)   # fitted scan pupil attribute
    """


@schema
class ScanPupil(ScanBehaviorTraceBase, dj.Lookup):
    definition = """
    -> pipe_eye.FittedPupil
    -> ScanPupilType
    """

    @row_property
    def times_trace(self):
        # times of eye trace on behavior clocks
        times = (pipe_eye.Eye & self).fetch1("eye_time")

        # fetch trace based on pupil type and attribute
        pupil_type, pupil_attr = self.fetch1("pupil_type", "pupil_attribute")

        if pupil_type == "circle":
            # fitted pupil circle
            fits = pipe_eye.FittedPupil.Circle & self

            if pupil_attr == "radius":
                # fitted circle radius
                trace = fits.fetch("radius", order_by="frame_id")

            elif pupil_attr in ["center_x", "center_y"]:
                # fitted circle center
                traces = fits.fetch("center", order_by="frame_id")

                if pupil_attr == "center_x":
                    trace = np.array([np.nan if t is None else t[0] for t in traces])
                else:
                    trace = np.array([np.nan if t is None else t[1] for t in traces])

            else:
                # other fitted circle attributes not implemented
                raise NotImplementedError()

        else:
            # other types not implemented
            raise NotImplementedError()

        return times, trace


@schema
class ScanTreadmill(ScanBehaviorTraceBase, dj.Lookup):
    definition = """
    -> pipe_tread.Treadmill
    """

    @row_property
    def times_trace(self):
        times, trace = (pipe_tread.Treadmill & self).fetch1("treadmill_time", "treadmill_vel")
        return times, trace


# -- Trace Link --


@link(schema)
class TraceLink:
    links = [MesoActivity, ScanPupil, ScanTreadmill]
    name = "trace"
    comment = "recording trace"


# -- Computed Trace --


@schema
class Trace(TraceBase, dj.Computed):
    definition = """
    -> TraceLink
    ---
    trials              : int unsigned      # number of trials
    """

    def make(self, key):
        trials = (TraceLink & key).link.trials
        key["trials"] = len(trials)
        self.insert1(key)

    @row_property
    def times_trace(self):
        times, trace = (TraceLink & self).link.times_trace
        return times, trace

    @row_property
    def trials(self):
        n = self.fetch1("trials")
        trials = (TraceLink & self).link.trials

        if n != len(trials):
            raise MissingError("Trials are missing")

        return trials

    @row_property
    def trial_flips(self):
        n = self.fetch1("trials")
        for trial, flips in (TraceLink & self).link.trial_flips:

            if not (np.diff(flips) > 0).all():
                raise ValueError("Flips do not monotonically increase.")

            yield trial, flips
            n -= 1

        if n == 0:
            return
        elif n > 0:
            raise MissingError("Missing trials.")
        else:
            raise RestrictionError("Extra trials received.")
