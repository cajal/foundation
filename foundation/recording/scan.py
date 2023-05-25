import numpy as np
from djutils import merge
from foundation.virtual import scan, stimulus
from foundation.virtual.bridge import pipe_stim, pipe_exp, pipe_shared
from foundation.recording.trial import Trial, TrialBounds, TrialVideo, TrialSet, TrialFilterSet
from foundation.recording.trace import Trace, TraceSet, TraceFilterSet
from foundation.schemas import recording as schema


@schema.computed
class ScanTrials:
    definition = """
    -> pipe_exp.Scan
    -> scan.FilteredTrials.proj(scan_filterset_id="trial_filterset_id")
    -> TrialFilterSet
    ---
    -> TrialSet
    """

    def make(self, key):
        from foundation.scan.trial import FilteredTrials, TrialSet as ScanTrialSet

        # filtered scan trials
        trials = FilteredTrials.proj(..., scan_filterset_id="trial_filterset_id") & key
        trials = ScanTrialSet & trials
        trials = merge(trials.members, Trial.ScanTrial)

        # filter trials
        trials = Trial & trials
        trials = (TrialFilterSet & key).filter(trials)

        # trial set
        trials = TrialSet.fill(trials, prompt=False)
        self.insert1(dict(key, **trials))


@schema.computed
class ScanPerspectives:
    definition = """
    -> pipe_exp.Scan
    -> pipe_shared.TrackingMethod
    -> scan.Scan
    -> TraceFilterSet
    ---
    -> TraceSet
    """

    def make(self, key):
        # scan pupil traces
        pupils = merge(
            scan.Scan & key,
            scan.PupilTrace & key,
            Trace.ScanPupil,
        )
        pupils &= [dict(pupil_type="center_x"), dict(pupil_type="center_y")]

        # filter traces
        traces = Trace & pupils
        traces = (TraceFilterSet & key).filter(traces)

        # trace set
        traces = TraceSet.fill(traces, prompt=False)
        self.insert1(dict(key, **traces))


@schema.computed
class ScanModulations:
    definition = """
    -> pipe_exp.Scan
    -> pipe_shared.TrackingMethod
    -> scan.Scan
    -> TraceFilterSet
    ---
    -> TraceSet
    """

    def make(self, key):
        # scan pupil trace
        pupil = merge(
            scan.Scan & key,
            scan.PupilTrace & key,
            Trace.ScanPupil,
        )
        pupil &= dict(pupil_type="radius")

        # scan treadmill trace
        tread = merge(
            scan.Scan & key,
            Trace.ScanTreadmill,
        )

        # filter traces
        traces = Trace & [pupil, tread]
        traces = (TraceFilterSet & key).filter(traces)

        # trace set
        traces = TraceSet.fill(traces, prompt=False)
        self.insert1(dict(key, **traces))


@schema.computed
class ScanUnits:
    definition = """
    -> pipe_exp.Scan
    -> pipe_shared.SpikeMethod
    -> scan.FilteredUnits
    -> TraceFilterSet
    ---
    -> TraceSet
    """

    def make(self, key):
        from foundation.scan.unit import FilteredUnits, UnitSet

        # filtered scan units
        units = FilteredUnits & key
        units = UnitSet & units
        units = merge(units.members, Trace.ScanUnit & key)

        # filter traces
        traces = Trace & units
        traces = (TraceFilterSet & key).filter(traces)

        # trace set
        traces = TraceSet.fill(traces, prompt=False)
        self.insert1(dict(key, **traces))


@schema.computed
class ScanVideoTiming:
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
