import numpy as np
from djutils import merge
from foundation.virtual import scan
from foundation.virtual.bridge import pipe_shared, pipe_stim
from foundation.recording.trial import TrialLink, TrialSet, TrialFilterSet
from foundation.recording.trace import TraceLink, TraceSet, TraceFilterSet
from foundation.schemas import recording as schema


@schema.computed
class FilteredScanTrials:
    definition = """
    -> scan.FilteredTrials.proj(scan_filters_id="trial_filters_id")
    -> TrialFilterSet
    ---
    -> TrialSet
    """

    def make(self, key):
        from foundation.scan.trial import FilteredTrials, TrialSet as ScanTrialSet

        # filtered scan trials
        trials = FilteredTrials.proj(..., scan_filters_id="trial_filters_id") & key
        trials = ScanTrialSet & trials
        trials = merge(trials.members, TrialLink.ScanTrial)

        # filter trials
        trials = TrialLink & trials
        trials = (TrialFilterSet & key).filter(trials)

        # trial set
        trials = TrialSet.fill(trials, prompt=False)
        self.insert1(dict(key, **trials))


@schema.computed
class FilteredScanPerspectives:
    definition = """
    -> scan.Scan
    -> pipe_shared.TrackingMethod
    -> TraceFilterSet
    ---
    -> TraceSet
    """

    def make(self, key):

        # scan pupil traces
        pupils = merge(
            scan.Scan & key,
            scan.PupilTrace & key,
            TraceLink.ScanPupil,
        )
        pupils &= [dict(pupil_type="center_x"), dict(pupil_type="center_y")]

        # filter traces
        traces = TraceLink & pupils
        traces = (TraceFilterSet & key).filter(traces)

        # trace set
        traces = TraceSet.fill(traces, prompt=False)
        self.insert1(dict(key, **traces))


@schema.computed
class FilteredScanModulations:
    definition = """
    -> scan.Scan
    -> pipe_shared.TrackingMethod
    -> TraceFilterSet
    ---
    -> TraceSet
    """

    def make(self, key):

        # scan pupil trace
        pupil = merge(
            scan.Scan & key,
            scan.PupilTrace & key,
            TraceLink.ScanPupil,
        )
        pupil &= dict(pupil_type="radius")

        # scan treadmill trace
        tread = merge(
            scan.Scan & key,
            TraceLink.ScanTreadmill,
        )

        # filter traces
        traces = TraceLink & [pupil, tread]
        traces = (TraceFilterSet & key).filter(traces)

        # trace set
        traces = TraceSet.fill(traces, prompt=False)
        self.insert1(dict(key, **traces))


@schema.computed
class FilteredScanUnits:
    definition = """
    -> scan.FilteredUnits
    -> pipe_shared.SpikeMethod
    -> TraceFilterSet
    ---
    -> TraceSet
    """

    def make(self, key):
        from foundation.scan.unit import FilteredUnits, UnitSet

        # filtered scan units
        units = FilteredUnits & key
        units = UnitSet & units
        units = merge(units.members, TraceLink.ScanUnit & key)

        # filter traces
        traces = TraceLink & units
        traces = (TraceFilterSet & key).filter(traces)

        # trace set
        traces = TraceSet.fill(traces, prompt=False)
        self.insert1(dict(key, **traces))
