import datajoint as dj
from djutils import merge, skip_missing
from foundation.recording import trial, trace
from foundation.scan import (
    timing as scan_timing,
    pupil as scan_pupil,
    trial as scan_trial,
    unit as scan_unit,
)
from foundation.schemas.pipeline import pipe_shared, pipe_stim
from foundation.schemas import recording as schema


def populate_scan(
    animal_id,
    session,
    scan_idx,
    tracking_method=2,
    spike_method=6,
    scan_trial_filters_id="244cbe86ea99f4f640ea15e5292d17ac",
    unit_filters_id="858ecab885fd2e3ba2979f7c79683365",
    trial_filters_id="07170d5333b23f57c17027073647fd3d",
    trace_filters_id="d41d8cd98f00b204e9800998ecf8427e",
):
    scan_key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx)

    def populate(table, key):
        table.populate(key, display_progress=True, reserve_jobs=True)

    def insert(table, keys):
        table.insert(keys, ignore_extra_fields=True, skip_duplicates=True)

    # scan timing
    populate(scan_timing.Timing, scan_key)
    insert(trace.ScanTreadmill, scan_timing.Timing & scan_key)

    # scan pupil
    key = dict(scan_key, tracking_method=tracking_method)
    populate(scan_pupil.PupilTrace, key)
    populate(scan_pupil.PupilNans, key)
    insert(trace.ScanPupil, scan_pupil.PupilTrace & key)

    # scan units
    key = dict(scan_key, unit_filters_id=unit_filters_id)
    populate(scan_unit.FilteredUnits, key)

    units = scan_unit.UnitSet & (scan_unit.FilteredUnits & key)
    units = units.members * pipe_shared.SpikeMethod & dict(spike_method=spike_method)
    insert(trace.ScanResponse, units)

    # scan trials
    key = dict(scan_key, trial_filters_id=scan_trial_filters_id)
    populate(scan_trial.FilteredTrials, key)
    insert(trial.ScanTrial, pipe_stim.Trial & scan_key)

    # fill links
    trial.TrialLink.fill()
    trace.TraceLink.fill()

    # populate trials
    populate(trial.TrialBounds, {})
    populate(trial.TrialSamples, {})
    populate(trial.TrialVideo, {})
    populate(trial.VideoSamples, {})

    # recording trial set
    key = dict(scan_key, scan_trial_filters_id=scan_trial_filters_id, trial_filters_id=trial_filters_id)
    populate(ScanTrials, key)

    # recording trace sets
    key = dict(scan_key, spike_method=spike_method, unit_filters_id=unit_filters_id, trace_filters_id=trace_filters_id)
    populate(ScanResponses, key)

    key = dict(scan_key, tracking_method=tracking_method, trace_filters_id=trace_filters_id)
    populate(ScanPerspective, key)
    populate(ScanModulation, key)


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
            trace.TraceLink.ScanPupil,
        )
        pupils &= [dict(pupil_type="center_x"), dict(pupil_type="center_y")]

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
            trace.TraceLink.ScanPupil,
        )
        pupil &= dict(pupil_type="radius")

        # scan treadmill trace
        tread = merge(
            scan_timing.Timing & key,
            trace.TraceLink.ScanTreadmill,
        )

        # filter traces
        traces = trace.TraceLink & [pupil, tread]
        for filter_key in (trace.TraceFilterSet & key).members.fetch(dj.key, order_by="member_id"):
            traces = (trace.TraceFilterLink & filter_key).link.filter(traces)

        # trace set
        traces = trace.TraceSet.fill(traces, prompt=False)
        self.insert1(dict(key, **traces))
