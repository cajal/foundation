from djutils import keys, merge, cache_rowproperty
from foundation.virtual.bridge import pipe_stim, pipe_tread, pipe_shared
from foundation.virtual import scan, utility, recording


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
            recording.TrialFilterSet,
            recording.TraceFilterSet,
        ]

    def fill(self):
        from foundation.recording.trial import ScanTrial, Trial, TrialBounds, TrialSamples, TrialVideo
        from foundation.recording.trace import ScanPupil, ScanTreadmill, ScanUnit, Trace, TraceHomogeneous, TraceTrials
        from foundation.recording.scan import ScanTrials, ScanVisualPerspectives, ScanVisualModulations, ScanUnits

        # scan trials
        key = pipe_stim.Trial & self.key - ScanTrial
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
        ScanPupil.insert(key - ScanPupil, skip_duplicates=True, ignore_extra_fields=True)
        keys.append([Trace.ScanPupil, key])

        # scan treadmill
        key = merge(self.key, pipe_tread.Treadmill)
        ScanTreadmill.insert(key - ScanTreadmill, skip_duplicates=True, ignore_extra_fields=True)
        keys.append([Trace.ScanTreadmill, key])

        # scan units
        key = merge(self.key, scan.FilteredUnits, scan.UnitSet.Member)
        ScanUnit.insert(key - ScanUnit, skip_duplicates=True, ignore_extra_fields=True)
        keys.append([Trace.ScanUnit, key])

        # trace link
        Trace.fill()

        # computed trace
        key = Trace.proj() & [t & k for t, k in keys]
        TraceHomogeneous.populate(key, display_progress=True, reserve_jobs=True)
        TraceTrials.populate(key, display_progress=True, reserve_jobs=True)

        # filtered scan trials ansd traces
        ScanTrials.populate(self.key, display_progress=True, reserve_jobs=True)
        ScanVisualPerspectives.populate(self.key, display_progress=True, reserve_jobs=True)
        ScanVisualModulations.populate(self.key, display_progress=True, reserve_jobs=True)
        ScanUnits.populate(self.key, display_progress=True, reserve_jobs=True)


@keys
class ScanVideoCache:
    """Cache resampled scan trial videos"""

    @property
    def key_list(self):
        return [
            recording.ScanTrials,
            utility.Rate,
        ]

    def fill(self):
        from foundation.recording.cache import ResampledVideo

        trials = recording.TrialSet.Member & (recording.ScanTrials & self.key)
        ResampledVideo.populate(trials, self.key, display_progress=True, reserve_jobs=True)


@keys
class ScanPerspectiveCache:
    """Cache resampled scan behavior traces"""

    @property
    def key_list(self):
        return [
            recording.ScanVisualPerspectives,
            recording.ScanTrials,
            utility.Resample,
            utility.Offset,
            utility.Rate,
        ]

    def fill(self):
        from foundation.recording.compute import TraceResampling
        from foundation.recording.cache import ResampledTraces

        # all keys
        key = self.key * recording.ScanVisualPerspectives * recording.ScanTrials * recording.TrialSet.Member

        # break up keys into TraceResampling groups
        for _key in (recording.TraceSet * utility.Rate * utility.Offset * utility.Resample & key).proj():

            # populate with TraceResampling cacheing
            with cache_rowproperty(TraceResampling):
                ResampledTraces.populate(_key, key, display_progress=True, reserve_jobs=True)


@keys
class ScanModulationCache:
    """Cache resampled scan behavior traces"""

    @property
    def key_list(self):
        return [
            recording.ScanVisualModulations,
            recording.ScanTrials,
            utility.Resample,
            utility.Offset,
            utility.Rate,
        ]

    def fill(self):
        from foundation.recording.compute import TraceResampling
        from foundation.recording.cache import ResampledTraces

        # all keys
        key = self.key * recording.ScanVisualModulations * recording.ScanTrials * recording.TrialSet.Member

        # break up keys into TraceResampling groups
        for _key in (recording.TraceSet * utility.Rate * utility.Offset * utility.Resample & key).proj():

            # populate with TraceResampling cacheing
            with cache_rowproperty(TraceResampling):
                ResampledTraces.populate(_key, key, display_progress=True, reserve_jobs=True)


@keys
class ScanUnitCache:
    """Cache resampled scan unit traces"""

    @property
    def key_list(self):
        return [
            recording.ScanUnits,
            recording.ScanTrials,
            utility.Resample,
            utility.Offset,
            utility.Rate,
        ]

    def fill(self):
        from foundation.recording.compute import TraceResampling
        from foundation.recording.cache import ResampledTraces

        # all keys
        key = self.key * recording.ScanUnits * recording.ScanTrials * recording.TrialSet.Member

        # break up keys into TraceResampling groups
        for _key in (recording.TraceSet * utility.Rate * utility.Offset * utility.Resample & key).proj():

            # populate with TraceResampling cacheing
            with cache_rowproperty(TraceResampling):
                ResampledTraces.populate(_key, key, display_progress=True, reserve_jobs=True)


@keys
class ScanPerspectiveSummary:
    """Summary of scan perspective traces"""

    @property
    def key_list(self):
        return [
            recording.ScanVisualPerspectives,
            recording.ScanTrials,
            utility.Summary,
            utility.Resample,
            utility.Offset,
            utility.Rate,
        ]

    def fill(self):
        from foundation.recording.trace import TraceSummary

        # summary keys
        key = (recording.TraceSet.Member * recording.TrialSet * self.key).proj()
        key &= recording.ScanTrials * recording.ScanVisualPerspectives & self.key

        # populate summary
        TraceSummary.populate(key, display_progress=True, reserve_jobs=True)


@keys
class ScanModulationSummary:
    """Summary of scan modulation traces"""

    @property
    def key_list(self):
        return [
            recording.ScanVisualModulations,
            recording.ScanTrials,
            utility.Summary,
            utility.Resample,
            utility.Offset,
            utility.Rate,
        ]

    def fill(self):
        from foundation.recording.trace import TraceSummary

        # summary keys
        key = (recording.TraceSet.Member * recording.TrialSet * self.key).proj()
        key &= recording.ScanTrials * recording.ScanVisualModulations & self.key

        # populate summary
        TraceSummary.populate(key, display_progress=True, reserve_jobs=True)


@keys
class ScanUnitSummary:
    """Summary of scan unit traces"""

    @property
    def key_list(self):
        return [
            recording.ScanUnits,
            recording.ScanTrials,
            utility.Summary,
            utility.Resample,
            utility.Offset,
            utility.Rate,
        ]

    def fill(self):
        from foundation.recording.trace import TraceSummary

        # summary keys
        key = (recording.TraceSet.Member * recording.TrialSet * self.key).proj()
        key &= recording.ScanTrials * recording.ScanUnits & self.key

        # populate summary
        TraceSummary.populate(key, display_progress=True, reserve_jobs=True)
