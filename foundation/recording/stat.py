import numpy as np
import datajoint as dj
from djutils import merge, skip_missing
from operator import add
from functools import reduce
from foundation.utility import stat
from foundation.recording import trial, trace
from foundation.schemas import recording as schema


@schema
class TraceSummary(dj.Computed):
    definition = """
    -> trace.TraceSamples
    -> trial.TrialSet
    -> stat.SummaryLink
    ---
    summary         : float         # summary statistic
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
        # trace samples
        df = (trace.TraceSamples & key).trials

        # trial set
        trials = (trial.TrialSet & key).members.fetch("trial_id", order_by="member_id")

        # trial set samples
        samples = np.concatenate(df.loc[trials].trace.values)

        # summary statistic
        key["summary"] = (stat.SummaryLink & key).link.stat(samples)
        self.insert1(key)
