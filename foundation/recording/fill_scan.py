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


# @keys
# class ScanTrials:
#     """Scan Trials"""

#     @property
#     def keys(self):
#         return [
#             recording.ScanTrials,
#             utility.Rate,
#             utility.Resize,
#             utility.Resolution,
#         ]

#     def fill(self):
#         from foundation.recording.trial import TrialSet, TrialSamples, TrialVideo
#         from foundation.recording.cache import ResampledTrial
#         from foundation.stimulus.cache import ResizedVideo

#         for key in self.key:

#             with cache_rowproperty():
#                 # trials
#                 trials = recording.ScanTrials & key
#                 trials = (TrialSet & trials).members

#                 # trial samples
#                 TrialSamples.populate(trials, key, display_progress=True, reserve_jobs=True)

#                 # resampled trial
#                 ResampledTrial.populate(trials, key, display_progress=True, reserve_jobs=True)

#                 # resized video
#                 videos = merge(trials, TrialVideo)
#                 ResizedVideo.populate(videos, key, display_progress=True, reserve_jobs=True)


# class _ScanTraces:
#     """Scan Traces"""

#     @property
#     def traces(self):
#         raise NotImplementedError()

#     @property
#     def keys(self):
#         return [
#             self.traces,
#             recording.ScanTrials,
#             utility.Standardize,
#             utility.Resample,
#             utility.Offset,
#             utility.Rate,
#         ]

#     def fill(self):
#         from foundation.recording.trial import TrialSet
#         from foundation.recording.trace import TraceSet, TraceSummary
#         from foundation.recording.cache import ResampledTraces
#         from foundation.utility.standardize import Standardize

#         for key in self.key:

#             with cache_rowproperty():
#                 # traces
#                 traces = self.traces & key
#                 traces = (TraceSet & traces).members

#                 # trials
#                 trials = recording.ScanTrials & key
#                 trials = (TrialSet & trials).members

#                 # stats
#                 stats = (Standardize & key).link.summary_ids
#                 stats = [{"summary_id": _} for _ in stats]

#                 # trace summary
#                 TraceSummary.populate(traces, trials, stats, key, display_progress=True, reserve_jobs=True)

#                 # resampled traces
#                 ResampledTraces.populate(traces, trials, key, display_progress=True, reserve_jobs=True)


# @keys
# class ScanUnits(_ScanTraces):
#     """Scan Units"""

#     @property
#     def traces(self):
#         return recording.ScanUnits


# @keys
# class ScanVisualPerspectives(_ScanTraces):
#     """Scan Visual Perspectives"""

#     @property
#     def traces(self):
#         return recording.ScanVisualPerspectives


# @keys
# class ScanVisualModulations(_ScanTraces):
#     """Scan Visual Modulations"""

#     @property
#     def traces(self):
#         return recording.ScanVisualModulations


# @keys
# class ScanUnitsVisualMeasure:
#     """Scan Units -- Visual Measure"""

#     @property
#     def keys(self):
#         return [
#             recording.ScanUnits,
#             recording.TrialFilterSet,
#             stimulus.VideoSet,
#             utility.Resample,
#             utility.Offset,
#             utility.Rate,
#             utility.Measure,
#             utility.Burnin,
#         ]

#     def fill(self):
#         from foundation.recording.trace import TraceSet
#         from foundation.recording.visual import VisualMeasure

#         for key in self.key:

#             with cache_rowproperty():
#                 # unit traces
#                 traces = recording.ScanUnits & key
#                 traces = (TraceSet & traces).members

#                 # visual measure
#                 VisualMeasure.populate(traces, key, display_progress=True, reserve_jobs=True)
