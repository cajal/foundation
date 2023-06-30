from djutils import rowproperty
from foundation.recording.trial import Trial, TrialSet
from foundation.schemas import recording as schema


# ----------------------------- Tier -----------------------------

# -- Tier Interface --


class TierType:
    """Trial Tier"""

    @rowproperty
    def tier(self):
        """
        Returns
        -------
        foundation.recording.compute_tier.Tier (row)
            trial tier
        """
        raise NotImplementedError()


# -- Tier Types --


@schema.lookup
class RandomSplit(TierType):
    definition = """
    fraction        : decimal(6, 6)     # split fraction
    seed            : int unsigned      # split seed
    """

    @rowproperty
    def tier(self):
        from foundation.recording.compute_tier import RandomSplit

        return RandomSplit & self


# --- Tier --


@schema.link
class Tier:
    links = [RandomSplit]
    name = "tier"
    comment = "trial tier"


# --- Computed Tier --


@schema.computed
class TrialTier:
    definition = """
    -> Tier
    -> TrialSet
    -> Trial
    ---
    tier_index      : int unsigned  # trial tier index
    """

    @property
    def key_source(self):
        return (TrialSet * Tier).proj()

    def make(self, key):
        # trial set
        trials = Trial & (TrialSet & key).members

        # trial tiers
        tiers = (Tier & key).link.tier.split(trials=trials)

        for index, trials in enumerate(tiers):

            # trial tier keys
            keys = [dict(key, tier_index=index, **trial) for trial in trials.fetch("KEY")]

            # insert
            self.insert(keys)
