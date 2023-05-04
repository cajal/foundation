from djutils import keys
from foundation.utility.resample import RateLink, OffsetLink, ResampleLink
from foundation.utility.stat import SummaryLink
from foundation.recording.scan import FilteredScanTrials, FilteredScanUnits
from foundation.recording.stat import TraceSummary
from foundation.recording.trial import TrialLink, TrialSet
from foundation.recording.trace import TraceLink, TraceSet


@keys
class ScanUnitSummary:
    keys = [FilteredScanTrials, FilteredScanUnits, RateLink, OffsetLink, ResampleLink, SummaryLink]

    def fill(self):
        trace_keys = TraceSet.Member & (FilteredScanUnits & self.key)
        trial_keys = TrialSet & (FilteredScanTrials & self.key)

        TraceSummary.populate(self.key, trial_keys, trace_keys, reserve_jobs=True, display_progress=True)
