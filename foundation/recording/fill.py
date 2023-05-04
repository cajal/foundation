from djutils import keys
from foundation.schemas.pipeline import pipe_stim
from foundation.utility.resample import RateLink, OffsetLink, ResampleLink
from foundation.utility.stat import SummaryLink
from foundation.scan.experiment import Scan
from foundation.recording.scan import FilteredScanTrials, FilteredScanUnits
from foundation.recording.stat import TraceSummary
from foundation.recording.trial import ScanTrial, TrialLink, TrialSet, TrialBounds, TrialVideo
from foundation.recording.trace import TraceLink, TraceSet


@keys
class ScanTrials:
    keys = [Scan]

    def fill(self):
        # trial keys
        key = pipe_stim.Trial & self.key
        ScanTrial.insert(key.proj(), skip_duplicates=True)

        # trial link
        TrialLink.fill()

        # computed trial
        key = TrialLink.ScanTrial & key
        TrialBounds.populate(key, display_progress=True, reserve_jobs=True)
        TrialVideo.populate(key, display_progress=True, reserve_jobs=True)


@keys
class ScanUnitSummary:
    keys = [FilteredScanTrials, FilteredScanUnits, RateLink, OffsetLink, ResampleLink, SummaryLink]

    def fill(self):
        trace_keys = TraceSet.Member & (FilteredScanUnits & self.key)
        trial_keys = TrialSet & (FilteredScanTrials & self.key)

        TraceSummary.populate(self.key, trial_keys, trace_keys, reserve_jobs=True, display_progress=True)
