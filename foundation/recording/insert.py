from djutils import keys, merge
from foundation.utils import multiprocess
from foundation.virtual.bridge import pipe_stim, pipe_tread, pipe_shared
from foundation.virtual import scan, utility
from foundation.recording.trial import (
    ScanTrial,
    TrialLink,
    TrialSet,
    TrialFilterSet,
    TrialBounds,
    TrialVideo,
)
from foundation.recording.trace import (
    ScanUnit,
    ScanPupil,
    ScanTreadmill,
    TraceLink,
    TraceSet,
    TraceFilterSet,
    TraceHomogeneous,
    TraceTrials,
    TraceSummary,
)
from foundation.recording.scan import (
    ScanTrials,
    ScanPerspectives,
    ScanModulations,
    ScanUnits,
)
from foundation.recording.cache import ResampledVideo, ResampledTraces


@keys
class Scan:
    """Scan recording"""

    @property
    def key_list(self):
        return [
            scan.Scan,
            pipe_shared.TrackingMethod,
            pipe_shared.SpikeMethod,
            scan.FilteredTrials.proj(scan_filters_id="trial_filters_id"),
            scan.FilteredUnits,
            TrialFilterSet,
            TraceFilterSet,
        ]

    def fill(self):
        # scan trials
        key = pipe_stim.Trial & self.key
        ScanTrial.insert(key, skip_duplicates=True, ignore_extra_fields=True)

        # trial link
        TrialLink.fill()

        # computed trial
        key = TrialLink.ScanTrial & self.key
        TrialBounds.populate(key, display_progress=True, reserve_jobs=True)
        TrialVideo.populate(key, display_progress=True, reserve_jobs=True)

        # trace keys
        keys = []

        # scan pupils
        key = merge(self.key, scan.PupilTrace)
        ScanPupil.insert(key, skip_duplicates=True, ignore_extra_fields=True)
        keys.append([TraceLink.ScanPupil, key])

        # scan treadmill
        key = merge(self.key, pipe_tread.Treadmill)
        ScanTreadmill.insert(key, skip_duplicates=True, ignore_extra_fields=True)
        keys.append([TraceLink.ScanTreadmill, key])

        # scan units
        key = merge(self.key, scan.FilteredUnits)
        key = scan.UnitSet.Unit & key
        ScanUnit.insert(key * self.key, skip_duplicates=True, ignore_extra_fields=True)
        keys.append([TraceLink.ScanUnit, key])

        # trace link
        TraceLink.fill()

        # computed trace
        key = TraceLink.proj() & [t & k for t, k in keys]
        TraceHomogeneous.populate(key, display_progress=True, reserve_jobs=True)
        TraceTrials.populate(key, display_progress=True, reserve_jobs=True)

        # filtered scan trials ansd traces
        ScanTrials.populate(self.key, display_progress=True, reserve_jobs=True)
        ScanPerspectives.populate(self.key, display_progress=True, reserve_jobs=True)
        ScanModulations.populate(self.key, display_progress=True, reserve_jobs=True)
        ScanUnits.populate(self.key, display_progress=True, reserve_jobs=True)


@keys
class ScanUnitSummary:
    """Summary of scan unit traces"""

    @property
    def key_list(self):
        return [
            ScanTrials,
            ScanUnits,
            utility.RateLink,
            utility.OffsetLink,
            utility.ResampleLink,
            utility.SummaryLink,
        ]

    def fill(self):
        key = TraceSet.Link * TrialSet.proj() * ScanTrials * ScanUnits * self.key
        TraceSummary.populate(key, display_progress=True, reserve_jobs=True)


@keys
class ScanBehaviorSummary:
    """Summary of scan behavior traces"""

    @property
    def key_list(self):
        return [
            ScanTrials,
            ScanPerspectives,
            ScanModulations,
            utility.RateLink,
            utility.OffsetLink,
            utility.ResampleLink,
            utility.SummaryLink,
        ]

    def fill(self):
        key = [
            TraceSet.Link * TrialSet.proj() * ScanTrials * ScanPerspectives * self.key,
            TraceSet.Link * TrialSet.proj() * ScanTrials * ScanModulations * self.key,
        ]
        TraceSummary.populate(key, display_progress=True, reserve_jobs=True)


@keys
class ScanVideoCache:
    """Cache resampled scan trial videos"""

    @property
    def key_list(self):
        return [
            ScanTrials,
            utility.RateLink,
        ]

    def fill(self):
        trials = TrialSet.Link & (ScanTrials & self.key)
        ResampledVideo.populate(trials, self.key, display_progress=True, reserve_jobs=True)


@keys
class ScanUnitCache:
    """Cache resampled scan unit traces"""

    @property
    def key_list(self):
        return [
            ScanTrials,
            ScanUnits,
            utility.RateLink,
            utility.OffsetLink,
            utility.ResampleLink,
        ]

    def fill(self, processes=1):
        trials = TrialSet.Link & (ScanTrials & self.key)
        trials = trials.proj()

        traces = TraceSet & (ScanUnits & self.key)
        traces = traces.proj()

        with multiprocess(processes):
            ResampledTraces.populate(trials, traces, self.key, display_progress=True, reserve_jobs=True)


@keys
class ScanBehaviorCache:
    """Cache resampled scan unit traces"""

    @property
    def key_list(self):
        return [
            ScanTrials,
            ScanPerspectives,
            ScanModulations,
            utility.RateLink,
            utility.OffsetLink,
            utility.ResampleLink,
        ]

    def fill(self, processes=1):
        trials = TrialSet.Link & (ScanTrials & self.key)
        trials = trials.proj()

        traces = TraceSet & [(ScanPerspectives & self.key), (ScanModulations & self.key)]
        traces = traces.proj()

        with multiprocess(processes):
            ResampledTraces.populate(trials, traces, self.key, display_progress=True, reserve_jobs=True)
