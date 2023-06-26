from djutils import keys, merge, cache_rowproperty
from foundation.virtual.bridge import pipe_fuse, pipe_shared
from foundation.virtual import scan, utility, recording


@keys
class Scan:
    """Scan Dataset"""

    @property
    def key_list(self):
        return [
            scan.Scan,
            pipe_fuse.ScanDone,
            pipe_shared.TrackingMethod & scan.PupilTrace,
            recording.TrialFilterSet,
            recording.TraceFilterSet,
        ]

    def fill(self):
        from foundation.recording.scan import (
            ScanRecording,
            ScanVideoTimeScale,
            ScanTrials,
            ScanUnits,
            ScanVisualPerspectives,
            ScanVisualModulations,
        )
        from foundation.recording.trial import TrialSet, TrialBounds, TrialVideo
        from foundation.recording.trace import TraceSet, TraceHomogeneous, TraceTrials

        # scan recording
        ScanRecording.populate(self.key, display_progress=True, reserve_jobs=True)
        ScanVideoTimeScale.populate(self.key, display_progress=True, reserve_jobs=True)

        for key in self.key:
            # trials
            trials = ScanRecording & key
            if trials:
                trials = (TrialSet & trials).members
            else:
                continue

            # compute trials
            TrialBounds.populate(trials, display_progress=True, reserve_jobs=True)
            TrialVideo.populate(trials, display_progress=True, reserve_jobs=True)

        # scan subsets
        for table in [ScanTrials, ScanUnits, ScanVisualPerspectives, ScanVisualModulations]:
            table.populate(self.key, display_progress=True, reserve_jobs=True)

        # scan traces
        for table in [ScanVisualPerspectives, ScanVisualModulations, ScanUnits]:

            for key in self.key:
                # traces
                traces = table & key
                if traces:
                    traces = (TraceSet & traces).members
                else:
                    continue

                # compute traces
                TraceHomogeneous.populate(traces, display_progress=True, reserve_jobs=True)
                TraceTrials.populate(traces, display_progress=True, reserve_jobs=True)


@keys
class ScanTrials:
    """Scan Trials"""

    @property
    def key_list(self):
        return [
            recording.ScanTrials,
            utility.Rate,
            utility.Resize,
            utility.Resolution,
        ]

    def fill(self):
        from foundation.recording.trial import TrialSet, TrialSamples, TrialVideo
        from foundation.recording.cache import ResampledTrial
        from foundation.stimulus.cache import ResizedVideo

        for key in self.key:

            with cache_rowproperty():
                # trials
                trials = recording.ScanTrials & key
                trials = (TrialSet & trials).members

                # trial samples
                TrialSamples.populate(trials, key, display_progress=True, reserve_jobs=True)

                # resampled trial
                ResampledTrial.populate(trials, key, display_progress=True, reserve_jobs=True)

                # resized video
                videos = merge(trials, TrialVideo)
                ResizedVideo.populate(videos, key, display_progress=True, reserve_jobs=True)


class ScanTraces:
    """Scan Traces"""

    @property
    def traces(self):
        raise NotImplementedError()

    @property
    def key_list(self):
        return [
            self.traces,
            recording.ScanTrials,
            utility.Standardize,
            utility.Resample,
            utility.Offset,
            utility.Rate,
        ]

    def fill(self):
        from foundation.recording.trial import TrialSet
        from foundation.recording.trace import TraceSet, TraceSummary
        from foundation.recording.cache import ResampledTraces
        from foundation.utility.standardize import Standardize

        for key in self.key:

            with cache_rowproperty():
                # traces
                traces = self.traces & key
                traces = (TraceSet & traces).members

                # trials
                trials = recording.ScanTrials & key
                trials = (TrialSet & trials).members

                # trace summary stats
                summaries = (Standardize & key).link.summaries.proj()
                TraceSummary.populate(traces, trials, summaries, key, display_progress=True, reserve_jobs=True)

                # resampled trace
                ResampledTraces.populate(traces, trials, key, display_progress=True, reserve_jobs=True)


@keys
class ScanUnits(ScanTraces):
    """Scan Units"""

    @property
    def traces(self):
        return recording.ScanUnits


@keys
class ScanVisualPerspectives(ScanTraces):
    """Scan Visual Perspectives"""

    @property
    def traces(self):
        return recording.ScanVisualPerspectives


@keys
class ScanVisualModulations(ScanTraces):
    """Scan Visual Modulations"""

    @property
    def traces(self):
        return recording.ScanVisualModulations
