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
    -> trace.TraceTrials
    -> trial.TrialSet
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
        keys = reduce(add, [dj.U("trace_id", "trials_id") & key for key in keys])
        return keys * resample.RateLink * resample.OffsetLink * resample.ResampleLink * stat.SummaryLink

    def make(self, key):
        # trial set
        trials_id = key.pop("trials_id")
        trials_key = trial.TrialSet & {"trials_id": trials_id}

        # resample keys
        rate_key = resample.RateLink & key
        offset_key = resample.OffsetLink & key
        resample_key = resample.ResampleLink & key

        # resampled trace
        a = (trace.TraceTrials & key).trial_samples(trials_key, rate_key, offset_key, resample_key)
        a = np.concatenate(a)

        # summary statistic for non-nan values
        n = np.isnan(a)
        s = (stat.SummaryLink & key).link.stat(a[~n])

        # insert key
        self.insert1(dict(key, trials_id=trials_id, summary=s, samples=len(a), nans=n.sum()))
