from foundation.virtual import utility
from foundation.recording.trace import Trace
from foundation.recording.trial import TrialSet
from foundation.schemas import recording as schema


# ----------------------------- Statistic -----------------------------


@schema.computed
class TraceSummary:
    definition = """
    -> Trace
    -> TrialSet
    -> utility.Summary
    -> utility.Resample
    -> utility.Offset
    -> utility.Rate
    ---
    summary = NULL      : float     # summary statistic
    """

    def make(self, key):
        from foundation.recording.compute.stat import TraceSummary

        # summary statistic
        key["summary"] = (TraceSummary & key).summary

        # insert
        self.insert1(key)
