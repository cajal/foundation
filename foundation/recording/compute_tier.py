import numpy as np
from djutils import keys, rowmethod
from foundation.virtual import recording


# ----------------------------- Tier -----------------------------

# -- Tier Interface --


class TierType:
    """Trial Tier"""

    @rowmethod
    def split(self, trials):
        """
        Parameters
        ----------
        trials : foundation.recording.trial.Trial (rows)
            trials to split into tiers

        Yields
        ------
        foundation.recording.trial.Trial (rows)
            tier of trials
        """
        raise NotImplementedError()


# -- Tier Types --


@keys
class RandomSplit(TierType):
    """Random Split"""

    @property
    def keys(self):
        return [
            recording.RandomSplit,
        ]

    @rowmethod
    def split(self, trials):
        # ordered trial ids
        trial_ids = trials.fetch("trial_id", order_by="trial_id")

        # random split
        rng = np.random.default_rng(self.item["seed"])
        trials_a = rng.choice(trial_ids, size=round(len(trial_ids) * self.item["fraction"]), replace=False)
        trials_a = set(trials_a)
        trials_b = set(trial_ids) - (trials_a)

        # yield trial tiers
        yield trials & [{"trial_id": t} for t in trials_a]
        yield trials & [{"trial_id": t} for t in trials_b]
