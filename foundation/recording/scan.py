import numpy as np
from djutils import merge
from foundation.virtual import scan, stimulus
from foundation.virtual.bridge import pipe_stim, pipe_shared, pipe_fuse, pipe_tread
from foundation.recording.trial import (
    ScanTrial,
    Trial,
    TrialBounds,
    TrialVideo,
    TrialSet,
    TrialFilterSet,
)
from foundation.recording.trace import (
    ScanUnit,
    ScanPupil,
    ScanTreadmill,
    Trace,
    TraceSet,
    TraceFilterSet,
)
from foundation.schemas import recording as schema


@schema.computed
class ScanTrials:
    definition = """
    -> scan.Scan
    -> TrialFilterSet
    ---
    -> TrialSet
    """

    def make(self, key):
        # scan trials
        trials = (pipe_stim.Trial & key).proj()

        # insert trials
        ScanTrial.insert(trials - ScanTrial)
        Trial.fill()

        # trial keys
        trials = Trial & (Trial.ScanTrial & key)

        # filter trials
        trials = (TrialFilterSet & key).filter(trials)

        # trial set
        trials = TrialSet.fill(trials, prompt=False)

        # insert
        self.insert1(dict(key, **trials))


@schema.computed
class ScanUnits:
    definition = """
    -> scan.Scan
    -> pipe_fuse.ScanDone
    -> TraceFilterSet
    ---
    -> TraceSet
    """

    def make(self, key):
        # scan units
        units = pipe_fuse.Activity.Trace & key

        # insert traces
        ScanUnit.insert(units - ScanUnit, ignore_extra_fields=True)
        Trace.fill()

        # trace keys
        traces = Trace & (Trace.ScanUnit & key)

        # filter traces
        traces = (TraceFilterSet & key).filter(traces)

        # trace set
        traces = TraceSet.fill(traces, prompt=False)

        # insert
        self.insert1(dict(key, **traces))


@schema.computed
class ScanVisualPerspectives:
    definition = """
    -> scan.Scan
    -> pipe_shared.TrackingMethod
    ---
    -> TraceSet
    """

    def make(self, key):
        # scan pupil traces
        pupils = merge(scan.Scan & key, scan.PupilTrace & key).proj()
        pupils &= [dict(pupil_type="center_x"), dict(pupil_type="center_y")]

        # insert traces
        ScanPupil.insert(pupils - ScanPupil)
        Trace.fill()

        # trace keys
        traces = Trace.ScanPupil & pupils

        # trace set
        traces = TraceSet.fill(traces, prompt=False)

        # insert
        self.insert1(dict(key, **traces))


@schema.computed
class ScanVisualModulations:
    definition = """
    -> scan.Scan
    -> pipe_shared.TrackingMethod
    ---
    -> TraceSet
    """

    def make(self, key):
        # scan pupil trace
        pupil = merge(scan.Scan & key, scan.PupilTrace & key).proj() & dict(pupil_type="radius")

        # scan treadmill trace
        tread = merge(scan.Scan & key, pipe_tread.Treadmill).proj()

        # insert traces
        ScanPupil.insert(pupil - ScanPupil)
        ScanTreadmill.insert(tread - ScanTreadmill)
        Trace.fill()

        # trace keys
        traces = [Trace.ScanPupil & pupil, Trace.ScanTreadmill & tread]

        # trace set
        traces = TraceSet.fill(traces, prompt=False)

        # insert
        self.insert1(dict(key, **traces))


@schema.computed
class ScanVideoTimeScale:
    definition = """
    -> scan.Scan
    ---
    time_scale  : double    # video time scale
    """

    def make(self, key):
        # trials merged with video info
        trials = (pipe_stim.Trial & key).proj()
        trials = merge(trials, Trial.ScanTrial, TrialBounds, TrialVideo, stimulus.VideoInfo)

        # trial and video timing
        start, end, frames, period = trials.fetch("start", "end", "frames", "period")
        rperiod = (end - start) / (frames - 1)

        # median time scale
        key["time_scale"] = np.nanmedian(rperiod / period)
        self.insert1(key)
