import numpy as np
from djutils import keys, rowmethod
from foundation.virtual import recording


# ----------------------------- Tier -----------------------------

# -- Tier Base --


class Tier:
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
class RandomSplit:
    """Random Split"""

    @property
    def key_list(self):
        return [
            recording.RandomSplit,
        ]

    @rowmethod
    def split(self, trials):
        # ordered trial ids
        trial_ids = trials.fetch("trial_id", order_by="trial_id")

        # split fraction, split seed
        fraction, seed = self.key.fetch1("fraction", "seed")

        # random split
        rng = np.random.default_rng(seed)
        trials_a = rng.choice(trial_ids, size=round(len(trial_ids) * fraction), replace=False)
        trials_a = set(trials_a)
        trials_b = set(trial_ids) - (trials_a)

        # yield trial tiers
        yield trials & [{"trial_id": t} for t in trials_a]
        yield trials & [{"trial_id": t} for t in trials_b]
