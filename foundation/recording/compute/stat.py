import numpy as np
from djutils import keys, rowproperty
from foundation.virtual import utility, recording


# ----------------------------- Statistic -----------------------------


@keys
class TraceSummary:
    """Trace Summary"""

    @property
    def keys(self):
        return [
            recording.Trace,
            recording.TrialSet & "members > 0",
            utility.Summary,
            utility.Resample,
            utility.Offset,
            utility.Rate,
        ]

    @rowproperty
    def summary(self):
        """
        Returns
        -------
        float
            trace summary statistic
        """
        from foundation.utility.stat import Summary
        from foundation.recording.trial import TrialSet
        from foundation.recording.compute.resample import ResampledTrace

        # recording trials
        trial_ids = (TrialSet & self.item).members.fetch("trial_id", order_by="trialset_index")

        # resampled traces
        trials = (ResampledTrace & self.item).trials(trial_ids)
        trials = np.concatenate(list(trials))

        # summary statistic
        return (Summary & self.item).link.summary(trials)
