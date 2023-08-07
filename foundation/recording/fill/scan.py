from djutils import keys, merge
from foundation.virtual.bridge import pipe_fuse, pipe_shared
from foundation.virtual import scan, recording


@keys
class VisualScanRecording:
    """Visual Scan Recording"""

    @property
    def keys(self):
        return [
            scan.Scan,
            pipe_fuse.ScanDone,
            pipe_shared.TrackingMethod & scan.PupilTrace,
        ]

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
        scan.ScanUnits.populate(self.key, trace_filt, display_progress=True, reserve_jobs=True)
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

            # unit traces (shortcut)
            traces = scan.ScanUnits & key & trace_filt
            traces = (trace.TraceSet & traces).members

            # trials
            trials = (scan.ScanRecording & key).fetch1("trialset_id")

            # keys
            keys = traces.fetch("trace_id", order_by="trace_id", as_dict=True)
            keys = [dict(key, trialset_id=trials, homogeneous=True) for key in keys]

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
