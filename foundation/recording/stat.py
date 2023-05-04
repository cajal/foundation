import numpy as np
import datajoint as dj
from operator import add
from functools import reduce
from foundation.utility.resample import RateLink, OffsetLink, ResampleLink
from foundation.utility.stat import SummaryLink
from foundation.recording.trial import TrialLink, TrialSet
from foundation.recording.trace import TraceLink, TraceSet
from foundation.recording.resample import ResampleTraceTrials
from foundation.schemas import recording as schema


@schema.computed
class TraceSummary:
    definition = """
    -> TraceLink
    -> TrialSet
    -> RateLink
    -> OffsetLink
    -> ResampleLink
    -> SummaryLink
    ---
    summary = NULL  : float             # summary statistic
    samples         : int unsigned      # number of samples
    nans            : int unsigned      # number of nans
    """

    @property
    def scan_keys(self):
        from foundation.recording.scan import (
            ScanTrialSet,
            ScanUnitSet,
            ScanModulationSet,
            ScanPerspectiveSet,
        )

        return [
            TraceSet.Member * ScanUnitSet * ScanTrialSet,
            TraceSet.Member * ScanPerspectiveSet * ScanTrialSet,
            TraceSet.Member * ScanModulationSet * ScanTrialSet,
        ]

    @property
    def key_source(self):
        keys = self.scan_keys
        keys = reduce(add, [dj.U("trace_id", "trials_id") & key for key in keys])
        return keys * (RateLink * OffsetLink * ResampleLink * SummaryLink).proj()

    def make(self, key):
        # trial set
        trial_keys = TrialLink & (TrialSet & key).members

        # resampled trace
        a = (ResampleTraceTrials & key & trial_keys).samples
        a = np.concatenate(a)

        # summary statistic for non-nan values
        n = np.isnan(a)
        s = (SummaryLink & key).link.summary(a[~n])

        # insert key
        self.insert1(dict(key, summary=s, samples=len(a), nans=n.sum()))
