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
            recording.TrialFilterSet,
            recording.TraceFilterSet,
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

        # scan tables
        scan.ScanTrials.populate(self.key, display_progress=True, reserve_jobs=True)
        scan.ScanUnits.populate(self.key, display_progress=True, reserve_jobs=True)
        scan.ScanVisualPerspectives.populate(self.key, display_progress=True, reserve_jobs=True)
        scan.ScanVisualModulations.populate(self.key, display_progress=True, reserve_jobs=True)
        scan.ScanVideoTimeScale.populate(self.key, display_progress=True, reserve_jobs=True)

        for table in [scan.ScanVisualPerspectives, scan.ScanVisualModulations, scan.ScanUnits]:

            for key in self.key:
                # traces
                traces = table & key
                traces = (trace.TraceSet & traces).members

                # compute traces
                trace.TraceHomogeneous.populate(traces, display_progress=True, reserve_jobs=True)
                trace.TraceTrials.populate(traces, display_progress=True, reserve_jobs=True)
