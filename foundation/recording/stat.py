import numpy as np
import datajoint as dj
from operator import add
from functools import reduce
from foundation.recording import trial, trace
from foundation.utility import resample, stat
from foundation.schemas import recording as schema


@schema.computed
class TraceSummary:
    definition = """
    -> trial.TrialSet
    -> trace.TraceLink
    -> resample.RateLink
    -> resample.OffsetLink
    -> resample.ResampleLink
    -> stat.SummaryLink
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
            trace.TraceSet.Member * ScanUnitSet * ScanTrialSet,
            trace.TraceSet.Member * ScanPerspectiveSet * ScanTrialSet,
            trace.TraceSet.Member * ScanModulationSet * ScanTrialSet,
        ]

    @property
    def key_source(self):
        keys = self.scan_keys
        keys = reduce(add, [dj.U("trials_id", "trace_id") & key for key in keys])
        return keys * resample.RateLink * resample.OffsetLink * resample.ResampleLink * stat.SummaryLink

    def make(self, key):
        # trial set
        trial_links = trial.TrialLink & (trial.TrialSet & key).members

        # resampling method
        rate_link = resample.RateLink & key
        offset_link = resample.OffsetLink & key
        resample_link = resample.ResampleLink & key

        # resampled trace
        a = (trace.TraceLink & key).resampled_trials(trial_links, rate_link, offset_link, resample_link)
        a = np.concatenate(a)

        # summary statistic for non-nan values
        n = np.isnan(a)
        s = (stat.SummaryLink & key).link.summary(a[~n])

        # insert key
        self.insert1(dict(key, summary=s, samples=len(a), nans=n.sum()))
