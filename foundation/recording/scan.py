import numpy as np
from djutils import merge
from foundation.recording import trial, trace
from foundation.stimulus import video
from foundation.scan import (
    experiment as scan_exp,
    pupil as scan_pupil,
    trial as scan_trial,
    unit as scan_unit,
)
from foundation.schemas.pipeline import pipe_shared, pipe_stim
from foundation.schemas import recording as schema


@schema.computed
class ScanTrialSet:
    definition = """
    -> scan_trial.FilteredTrials.proj(scan_trial_filters_id="trial_filters_id")
    -> trial.TrialFilterSet
    ---
    -> trial.TrialSet
    """

    def make(self, key):
        # filtered scan trials
        trials = scan_trial.FilteredTrials.proj(..., scan_trial_filters_id="trial_filters_id") & key
        trials = scan_trial.TrialSet & trials
        trials = merge(trials.members, trial.TrialLink.ScanTrial)

        # filter trials
        trials = trial.TrialLink & trials
        trials = (trial.TrialFilterSet & key).filter(trials)

        # trial set
        trials = trial.TrialSet.fill(trials, prompt=False)
        self.insert1(dict(key, **trials))


class _TraceSet:
    """Scan Trace Set"""

    def fill_stats(self, trial_key={}, rate_key={}, offset_key={}, resample_key={}, summary_key={}):
        """
        Parameters
        ----------
        trial_key : datajoint.key -- foundation.recording.trial.TrialLink
            restriction
        rate_key : datajoint.key -- foundation.utility.resample.RateLink
            restriction
        offset_key : datajoint.key -- foundation.utility.resample.OffsetLink
            restriction
        resample_key : datajoint.key -- foundation.utility.resample.ResampleLink
            restriction
        summary_key : datajoint.key -- foundation.utility.stat.SummaryLink
            restriction
        """
        from foundation.recording.stat import TraceSummary
        from foundation.utility import resample, stat

        TraceSummary.populate(
            (trace.TraceSet.Member & self).proj(),
            (trial.TrialLink & trial_key).proj(),
            (resample.RateLink & rate_key).proj(),
            (resample.OffsetLink & offset_key).proj(),
            (resample.ResampleLink & resample_key).proj(),
            (stat.SummaryLink & summary_key).proj(),
            display_progress=True,
            reserve_jobs=True,
        )


@schema.computed
class ScanUnitSet(_TraceSet):
    definition = """
    -> scan_unit.FilteredUnits
    -> trace.TraceFilterSet
    -> pipe_shared.SpikeMethod
    ---
    -> trace.TraceSet
    """

    def make(self, key):
        # filtered scan units
        units = scan_unit.FilteredUnits & key
        units = scan_unit.UnitSet & units
        units = merge(units.members, trace.TraceLink.ScanUnit & key)

        # filter traces
        traces = trace.TraceLink & units
        traces = (trace.TraceFilterSet & key).filter(traces)

        # trace set
        traces = trace.TraceSet.fill(traces, prompt=False)
        self.insert1(dict(key, **traces))


@schema.computed
class ScanPerspectiveSet(_TraceSet):
    definition = """
    -> scan_exp.Scan
    -> pipe_shared.TrackingMethod
    -> trace.TraceFilterSet
    ---
    -> trace.TraceSet
    """

    def make(self, key):
        # scan pupil traces
        pupils = merge(
            scan_exp.Scan & key,
            scan_pupil.PupilTrace & key,
            trace.TraceLink.ScanPupil,
        )
        pupils &= [dict(pupil_type="center_x"), dict(pupil_type="center_y")]

        # filter traces
        traces = trace.TraceLink & pupils
        traces = (trace.TraceFilterSet & key).filter(traces)

        # trace set
        traces = trace.TraceSet.fill(traces, prompt=False)
        self.insert1(dict(key, **traces))


@schema.computed
class ScanModulationSet(_TraceSet):
    definition = """
    -> scan_exp.Scan
    -> pipe_shared.TrackingMethod
    -> trace.TraceFilterSet
    ---
    -> trace.TraceSet
    """

    def make(self, key):
        # scan pupil trace
        pupil = merge(
            scan_exp.Scan & key,
            scan_pupil.PupilTrace & key,
            trace.TraceLink.ScanPupil,
        )
        pupil &= dict(pupil_type="radius")

        # scan treadmill trace
        tread = merge(
            scan_exp.Scan & key,
            trace.TraceLink.ScanTreadmill,
        )

        # filter traces
        traces = trace.TraceLink & [pupil, tread]
        traces = (trace.TraceFilterSet & key).filter(traces)

        # trace set
        traces = trace.TraceSet.fill(traces, prompt=False)
        self.insert1(dict(key, **traces))
