from djutils import keys
from foundation.utility.resample import RateLink, OffsetLink, ResampleLink
from foundation.utility.stat import SummaryLink
from foundation.recording.scan import ScanTrialSet, ScanUnitSet
from foundation.recording.stat import TraceSummary
from foundation.recording.trial import TrialLink, TrialSet
from foundation.recording.trace import TraceLink, TraceSet


@keys
class ScanUnitSummary:
    keys = [ScanTrialSet, ScanUnitSet, RateLink, OffsetLink, ResampleLink, SummaryLink]

    def fill(self):
        trace_keys = TraceSet.Member & (ScanUnitSet & self.key)
        trial_keys = TrialSet & (ScanTrialSet & self.key)

        TraceSummary.populate(self.key, trial_keys, trace_keys, reserve_jobs=True, display_progress=True)
