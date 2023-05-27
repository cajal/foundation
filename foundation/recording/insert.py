from djutils import keys, merge, cache_rowproperty
from foundation.virtual.bridge import pipe_fuse, pipe_shared
from foundation.virtual import scan, utility, recording


@keys
class Scan:
    """Scan Recording"""

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
        from foundation.recording.scan import ScanTrials, ScanUnits, ScanVisualPerspectives, ScanVisualModulations
        from foundation.recording.trace import TraceSet, TraceHomogeneous, TraceTrials
        from foundation.recording.trial import TrialSet, TrialBounds

        # scan dataset
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

                # trials
                trials = TrialSet & (TraceTrials & traces)
                trials = trials.members

                # compute trials
                TrialBounds.populate(trials, display_progress=True, reserve_jobs=True)


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

    def fill(self, cache=True):
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

                if cache:
                    # resampled trial
                    ResampledTrial.populate(trials, key, display_progress=True, reserve_jobs=True)

                    # resized video
                    videos = merge(trials, TrialVideo)
                    ResizedVideo.populate(videos, key, display_progress=True, reserve_jobs=True)


@keys
class ScanVisualPerspectives:
    """Scan Visual Perspectives"""

    @property
    def key_list(self):
        return [
            recording.ScanTrials,
            recording.ScanVisualPerspectives,
            utility.Rate,
            utility.Offset,
            utility.Resample,
            utility.Standardize,
        ]

    def fill(self, cache=True):
        from foundation.recording.scan import ScanTrials, ScanVisualPerspectives
        from foundation.recording.trial import TrialSet
        from foundation.recording.trace import TraceSet, TraceSummary
        from foundation.recording.cache import ResampledTraces
        from foundation.utility.standardize import Standardize

        for key in self.key:

            with cache_rowproperty():
                # traces
                traces = ScanVisualPerspectives & key
                traces = (TraceSet & traces).members

                # trials
                trials = ScanTrials & key
                trials = (TrialSet & trials).members

                # trace summary stats
                summaries = (Standardize & key).link.summary_keys.proj()
                TraceSummary.populate(traces, trials, summaries, key, display_progress=True, reserve_jobs=True)

                if cache:
                    # resampled trace
                    ResampledTraces.populate(traces, trials, key, display_progress=True, reserve_jobs=True)


@keys
class ScanUnits:
    """Scan Units"""

    @property
    def key_list(self):
        return [
            recording.ScanTrials,
            recording.ScanUnits,
            utility.Rate,
            utility.Offset,
            utility.Resample,
            utility.Standardize,
        ]

    def fill(self, cache=True):
        from foundation.recording.scan import ScanTrials, ScanUnits
        from foundation.recording.trial import TrialSet
        from foundation.recording.trace import TraceSet, TraceSummary
        from foundation.recording.cache import ResampledTraces
        from foundation.utility.standardize import Standardize

        for key in self.key:

            with cache_rowproperty():
                # traces
                traces = ScanUnits & key
                traces = (TraceSet & traces).members

                # trials
                trials = ScanTrials & key
                trials = (TrialSet & trials).members

                # trace summary stats
                summaries = (Standardize & key).link.summary_keys.proj()
                TraceSummary.populate(traces, trials, summaries, key, display_progress=True, reserve_jobs=True)

                if cache:
                    # resampled trace
                    ResampledTraces.populate(traces, trials, key, display_progress=True, reserve_jobs=True)


@keys
class ScanVisualPerspectives:
    """Scan Visual Perspectives"""

    @property
    def key_list(self):
        return [
            recording.ScanTrials,
            recording.ScanVisualPerspectives,
            utility.Rate,
            utility.Offset,
            utility.Resample,
            utility.Standardize,
        ]

    def fill(self, cache=True):
        from foundation.recording.scan import ScanTrials, ScanVisualPerspectives
        from foundation.recording.trial import TrialSet
        from foundation.recording.trace import TraceSet, TraceSummary
        from foundation.recording.cache import ResampledTraces
        from foundation.utility.standardize import Standardize

        for key in self.key:

            with cache_rowproperty():
                # traces
                traces = ScanVisualPerspectives & key
                traces = (TraceSet & traces).members

                # trials
                trials = ScanTrials & key
                trials = (TrialSet & trials).members

                # trace summary stats
                summaries = (Standardize & key).link.summary_keys.proj()
                TraceSummary.populate(traces, trials, summaries, key, display_progress=True, reserve_jobs=True)

                if cache:
                    # resampled trace
                    ResampledTraces.populate(traces, trials, key, display_progress=True, reserve_jobs=True)


@keys
class ScanVisualModulations:
    """Scan Visual Modulations"""

    @property
    def key_list(self):
        return [
            recording.ScanTrials,
            recording.ScanVisualModulations,
            utility.Rate,
            utility.Offset,
            utility.Resample,
            utility.Standardize,
        ]

    def fill(self, cache=True):
        from foundation.recording.scan import ScanTrials, ScanVisualModulations
        from foundation.recording.trial import TrialSet
        from foundation.recording.trace import TraceSet, TraceSummary
        from foundation.recording.cache import ResampledTraces
        from foundation.utility.standardize import Standardize

        for key in self.key:

            with cache_rowproperty():
                # traces
                traces = ScanVisualModulations & key
                traces = (TraceSet & traces).members

                # trials
                trials = ScanTrials & key
                trials = (TrialSet & trials).members

                # trace summary stats
                summaries = (Standardize & key).link.summary_keys.proj()
                TraceSummary.populate(traces, trials, summaries, key, display_progress=True, reserve_jobs=True)

                if cache:
                    # resampled trace
                    ResampledTraces.populate(traces, trials, key, display_progress=True, reserve_jobs=True)
