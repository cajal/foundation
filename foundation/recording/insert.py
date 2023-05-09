from djutils import keys, merge, cache_rowproperty
from foundation.virtual.bridge import pipe_stim, pipe_tread, pipe_shared
from foundation.virtual import scan, utility
from foundation.recording.trial import (
    ScanTrial,
    Trial,
    TrialSet,
    TrialFilterSet,
    TrialBounds,
    TrialSamples,
    TrialVideo,
)
from foundation.recording.trace import (
    ScanUnit,
    ScanPupil,
    ScanTreadmill,
    Trace,
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
            pipe_shared.TrackingMethod & scan.PupilTrace,
            pipe_shared.SpikeMethod & "spike_method in (5, 6)",
            scan.FilteredTrials.proj(scan_filterset_id="trial_filterset_id"),
            scan.FilteredUnits,
            TrialFilterSet,
            TraceFilterSet,
        ]

    def fill(self):
        # scan trials
        key = pipe_stim.Trial & self.key
        ScanTrial.insert(key, skip_duplicates=True, ignore_extra_fields=True)

        # trial link
        Trial.fill()

        # computed trial
        key = Trial.ScanTrial & self.key
        TrialBounds.populate(key, display_progress=True, reserve_jobs=True)
        TrialSamples.populate(key, display_progress=True, reserve_jobs=True)
        TrialVideo.populate(key, display_progress=True, reserve_jobs=True)

        # trace keys
        keys = []

        # scan pupils
        key = merge(self.key, scan.PupilTrace)
        ScanPupil.insert(key, skip_duplicates=True, ignore_extra_fields=True)
        keys.append([Trace.ScanPupil, key])

        # scan treadmill
        key = merge(self.key, pipe_tread.Treadmill)
        ScanTreadmill.insert(key, skip_duplicates=True, ignore_extra_fields=True)
        keys.append([Trace.ScanTreadmill, key])

        # scan units
        key = merge(self.key, scan.FilteredUnits)
        key = scan.UnitSet.Member & key
        ScanUnit.insert(key * self.key, skip_duplicates=True, ignore_extra_fields=True)
        keys.append([Trace.ScanUnit, key])

        # trace link
        Trace.fill()

        # computed trace
        key = Trace.proj() & [t & k for t, k in keys]
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
            utility.Rate,
            utility.Offset,
            utility.Resample,
            utility.Summary,
        ]

    def fill(self):
        key = TraceSet.Member * TrialSet.proj() * ScanTrials * ScanUnits * self.key
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
            utility.Rate,
            utility.Offset,
            utility.Resample,
            utility.Summary,
        ]

    def fill(self):
        key = [
            TraceSet.Member * TrialSet.proj() * ScanTrials * ScanPerspectives * self.key,
            TraceSet.Member * TrialSet.proj() * ScanTrials * ScanModulations * self.key,
        ]
        TraceSummary.populate(key, display_progress=True, reserve_jobs=True)


@keys
class ScanVideoCache:
    """Cache resampled scan trial videos"""

    @property
    def key_list(self):
        return [
            ScanTrials,
            utility.Rate,
        ]

    def fill(self):
        trials = TrialSet.Member & (ScanTrials & self.key)
        ResampledVideo.populate(trials, self.key, display_progress=True, reserve_jobs=True)


@keys
class ScanUnitCache:
    """Cache resampled scan unit traces"""

    @property
    def key_list(self):
        return [
            ScanTrials,
            ScanUnits,
            utility.Rate,
            utility.Offset,
            utility.Resample,
        ]

    def fill(self):
        from foundation.recording.compute import TraceResampling

        # all keys
        key = ScanTrials * ScanUnits * self.key * TrialSet.Member

        # break up keys into TraceResampling groups
        for _key in (TraceSet * utility.Rate * utility.Offset * utility.Resample & key).proj():

            # populate with TraceResampling cacheing
            with cache_rowproperty(TraceResampling):
                ResampledTraces.populate(_key, key, display_progress=True, reserve_jobs=True)


@keys
class ScanBehaviorCache:
    """Cache resampled scan behavior traces"""

    @property
    def key_list(self):
        return [
            ScanTrials,
            ScanPerspectives,
            ScanModulations,
            utility.Rate,
            utility.Offset,
            utility.Resample,
        ]

    def fill(self):
        from foundation.recording.compute import TraceResampling

        for behavior in [ScanPerspectives, ScanModulations]:

            # all keys
            key = ScanTrials * behavior * self.key * TrialSet.Member

            # break up keys into TraceResampling groups
            for _key in (TraceSet * utility.Rate * utility.Offset * utility.Resample & key).proj():

                # populate with TraceResampling cacheing
                with cache_rowproperty(TraceResampling):
                    ResampledTraces.populate(_key, key, display_progress=True, reserve_jobs=True)
