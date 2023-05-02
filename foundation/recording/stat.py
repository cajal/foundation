import numpy as np
import datajoint as dj
from operator import add
from functools import reduce
from foundation.recording import trial, trace
from foundation.utility import stat
from foundation.schemas import recording as schema


@schema.computed
class TraceSummary:
    definition = """
    -> trace.TraceSamples
    -> trial.TrialSet
    -> stat.SummaryLink
    ---
    summary = NULL  : float             # summary statistic
    samples         : int unsigned      # number of samples
    nans            : int unsigned      # number of nans
    """

    @property
    def key_source(self):
        from foundation.recording.scan import (
            ScanTrials,
            ScanResponses,
            ScanModulation,
            ScanPerspective,
        )

        keys = [
            trace.TraceSet.Member * ScanResponses * ScanTrials,
            trace.TraceSet.Member * ScanModulation * ScanTrials,
            trace.TraceSet.Member * ScanPerspective * ScanTrials,
        ]
        keys = reduce(add, [dj.U("trace_id", "trials_id") & key for key in keys])

        return keys * trace.TraceSamples.proj() * stat.SummaryLink.proj()

    def make(self, key):
        # trial set
        trials = (trial.TrialSet & key).members.fetch("trial_id", order_by="member_id")

        # trial set samples
        df = (trace.TraceSamples & key).samples.loc[trials]

        # sample values and nans
        a = np.concatenate(df.trace.values)
        n = np.isnan(a)

        # summary statistic for non-nan values
        summary = (stat.SummaryLink & key).link.stat(a[~n])

        # insert key
        self.insert1(dict(key, summary=summary, samples=len(a), nans=n.sum()))
