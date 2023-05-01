import datajoint as dj
from djutils import merge, skip_missing
from foundation.recording import trial, trace
from foundation.scan import (
    timing as scan_timing,
    pupil as scan_pupil,
    trial as scan_trial,
    unit as scan_unit,
)
from foundation.schemas.pipeline import pipe_shared
from foundation.schemas import recording as schema


@schema
class ScanTrials(dj.Computed):
    definition = """
    -> scan_trial.FilteredTrials.proj(scan_trial_filters_id='trial_filters_id')
    -> trial.TrialFilterSet
    ---
    -> trial.TrialSet
    """

    @skip_missing
    def make(self, key):
        # filtered scan trials
        trials = scan_trial.FilteredTrials.proj(..., scan_trial_filters_id="trial_filters_id") & key
        trials = scan_trial.TrialSet & trials
        trials = merge(trials.members, trial.TrialLink.ScanTrial)

        # filter trials
        trials = trial.TrialLink & trials
        for filter_key in (trial.TrialFilterSet & key).members.fetch(dj.key, order_by="member_id"):
            trials = (trial.TrialFilterLink & filter_key).link.filter(trials)

        # trial set
        trials = trial.TrialSet.fill(trials, prompt=False)
        self.insert1(dict(key, **trials))


@schema
class ScanResponses(dj.Computed):
    definition = """
    -> scan_unit.FilteredUnits
    -> trace.TraceFilterSet
    -> pipe_shared.SpikeMethod
    ---
    -> trace.TraceSet
    """

    @skip_missing
    def make(self, key):
        # filtered scan units
        units = scan_unit.FilteredUnits & key
        units = scan_unit.UnitSet & units
        units = merge(units.members, trace.TraceLink.ScanResponse & key)

        # filter traces
        traces = trace.TraceLink & units
        for filter_key in (trace.TraceFilterSet & key).members.fetch(dj.key, order_by="member_id"):
            traces = (trace.TraceFilterLink & filter_key).link.filter(traces)

        # trace set
        traces = trace.TraceSet.fill(traces, prompt=False)
        self.insert1(dict(key, **traces))


@schema
class ScanPerspective(dj.Computed):
    definition = """
    -> scan_timing.Timing
    -> pipe_shared.TrackingMethod
    -> trace.TraceFilterSet
    ---
    -> trace.TraceSet
    """

    @skip_missing
    def make(self, key):
        # scan pupil traces
        pupils = merge(
            scan_timing.Timing & key,
            scan_pupil.PupilTrace & key,
            trace.TraceLink.ScanPupil & key,
        ) & [dict(pupil_type="center_x"), dict(pupil_type="center_y")]

        # filter traces
        traces = trace.TraceLink & pupils
        for filter_key in (trace.TraceFilterSet & key).members.fetch(dj.key, order_by="member_id"):
            traces = (trace.TraceFilterLink & filter_key).link.filter(traces)

        # trace set
        traces = trace.TraceSet.fill(traces, prompt=False)
        self.insert1(dict(key, **traces))


@schema
class ScanModulation(dj.Computed):
    definition = """
    -> scan_timing.Timing
    -> pipe_shared.TrackingMethod
    -> trace.TraceFilterSet
    ---
    -> trace.TraceSet
    """

    @skip_missing
    def make(self, key):
        # scan pupil trace
        pupil = merge(
            scan_timing.Timing & key,
            scan_pupil.PupilTrace & key,
            trace.TraceLink.ScanPupil & key,
        ) & dict(pupil_type="radius")

        # scan treadmill trace
        tread = merge(
            scan_timing.Timing & key,
            trace.TraceLink.ScanTreadmill & key,
        )

        # filter traces
        traces = trace.TraceLink & [pupil, tread]
        for filter_key in (trace.TraceFilterSet & key).members.fetch(dj.key, order_by="member_id"):
            traces = (trace.TraceFilterLink & filter_key).link.filter(traces)

        # trace set
        traces = trace.TraceSet.fill(traces, prompt=False)
        self.insert1(dict(key, **traces))
