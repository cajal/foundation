from djutils import keys, merge, cache_rowproperty
from foundation.virtual.bridge import pipe_fuse, pipe_shared
from foundation.virtual import scan, recording, stimulus, utility


class _VisualScanRecording:
    """Visual Scan Recording -- Base"""

    @property
    def units_table(self):
        raise NotImplementedError()

    @property
    def units_homogeneous(self):
        raise NotImplementedError()

    def fill(self):
        from foundation.recording import trial, trace, scan

        # scan recording
        scan.ScanRecording.populate(self.key, display_progress=True, reserve_jobs=True)

        for key in self.key:
            # trials
            trials = scan.ScanRecording & key
            trials = (trial.TrialSet & trials).members

            # compute trials
            trial.TrialBounds.populate(trials, display_progress=True, reserve_jobs=True)
            trial.TrialVideo.populate(trials, display_progress=True, reserve_jobs=True)

        # scan time scale
        scan.ScanVideoTimeScale.populate(self.key, display_progress=True, reserve_jobs=True)

        # empty filter sets
        trial_filt = (recording.TrialFilterSet & "not members").proj()
        trace_filt = (recording.TraceFilterSet & "not members").proj()

        # scan sets
        scan.ScanTrials.populate(self.key, trial_filt, display_progress=True, reserve_jobs=True)
        self.units_table.populate(self.key, trace_filt, display_progress=True, reserve_jobs=True)
        scan.ScanVisualPerspectives.populate(self.key, display_progress=True, reserve_jobs=True)
        scan.ScanVisualModulations.populate(self.key, display_progress=True, reserve_jobs=True)
        scan.ScanVideoTimeScale.populate(self.key, display_progress=True, reserve_jobs=True)

        for key in self.key:

            # perspective and modulation traces
            for table in [scan.ScanVisualPerspectives, scan.ScanVisualModulations]:

                # traces
                traces = table & key
                traces = (trace.TraceSet & traces).members

                # compute perspective/modulation traces
                trace.TraceHomogeneous.populate(traces, display_progress=True, reserve_jobs=True)
                trace.TraceTrials.populate(traces, display_progress=True, reserve_jobs=True)

            # ---- shortcut for units ----

            # traces
            traces = self.units_table & key & trace_filt
            traces = (trace.TraceSet & traces).members

            # trials
            trials = (scan.ScanRecording & key).fetch1("trialset_id")

            # keys
            keys = traces.fetch("trace_id", order_by="trace_id", as_dict=True)
            keys = [dict(key, trialset_id=trials, homogeneous=self.units_homogeneous) for key in keys]

            # insert unit traces
            trace.TraceTrials.insert(
                keys,
                skip_duplicates=True,
                ignore_extra_fields=True,
                allow_direct_insert=True,
            )
            trace.TraceHomogeneous.insert(
                keys,
                skip_duplicates=True,
                allow_direct_insert=True,
                ignore_extra_fields=True,
            )


@keys
class VisualScanRecording(_VisualScanRecording):
    """Visual Scan Recording -- Activity"""

    @property
    def keys(self):
        return [
            scan.Scan,
            pipe_fuse.ScanDone,
            pipe_shared.TrackingMethod & scan.PupilTrace,
        ]

    @property
    def units_table(self):
        from foundation.recording.scan import ScanUnits

        return ScanUnits

    @property
    def units_homogeneous(self):
        return True


@keys
class VisualScanRawRecording(_VisualScanRecording):
    """Visual Scan Recording -- Fluorescence"""

    @property
    def keys(self):
        return [
            (scan.Scan * pipe_shared.PipelineVersion * pipe_shared.SegmentationMethod).proj()
            & pipe_fuse.ScanDone,
            pipe_shared.TrackingMethod & scan.PupilTrace,
        ]

    @property
    def units_table(self):
        from foundation.recording.scan import ScanUnitsRaw

        return ScanUnitsRaw

    @property
    def units_homogeneous(self):
        return False


@keys
class VisualScanMeasure:
    """Visual Scan Measure"""

    @property
    def keys(self):
        return [
            recording.ScanUnits,
            recording.TrialFilterSet,
            stimulus.VideoSet,
            utility.Resample,
            utility.Offset,
            utility.Rate,
            utility.Measure,
            utility.Burnin,
        ]

    def fill(self):
        from foundation.recording.trace import TraceSet
        from foundation.recording.visual import VisualMeasure

        for key in self.key:

            traces = recording.ScanUnits & key
            traces = (TraceSet & traces).members

            with cache_rowproperty():

                VisualMeasure.populate(key, traces, reserve_jobs=True, display_progress=True)


@keys
class VisualScanDirectionTuning:
    """Visual Scan Direction Tuning"""

    @property
    def keys(self):
        return [
            recording.ScanUnits,
            recording.TrialFilterSet,
            stimulus.VideoSet,
            utility.Offset,
            utility.Impulse,
            utility.Precision,
        ]

    def fill(self):
        from foundation.recording.trace import TraceSet
        from foundation.recording.visual import VisualDirectionTuning

        for key in self.key:

            traces = recording.ScanUnits & key
            traces = (TraceSet & traces).members

            with cache_rowproperty():

                VisualDirectionTuning.populate(key, traces, reserve_jobs=True, display_progress=True)


@keys
class VisualScanSpatialTuning:
    """Visual Scan Spatial Tuning"""

    @property
    def keys(self):
        return [
            recording.ScanUnits,
            recording.TrialFilterSet,
            stimulus.VideoSet,
            utility.Offset,
            utility.Impulse,
            utility.Resolution,
        ]

    def fill(self):
        from foundation.recording.trace import TraceSet
        from foundation.recording.visual import VisualSpatialTuning

        for key in self.key:

            traces = recording.ScanUnits & key
            traces = (TraceSet & traces).members

            with cache_rowproperty():

                VisualSpatialTuning.populate(key, traces, reserve_jobs=True, display_progress=True)
