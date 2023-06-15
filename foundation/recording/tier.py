from djutils import rowproperty
from foundation.recording.trial import Trial, TrialSet
from foundation.schemas import recording as schema


# ----------------------------- Tier -----------------------------

# -- Tier Base --


class _Tier:
    """Trial Tier"""

    @rowproperty
    def tier(self):
        """
        Returns
        -------
        foundation.recording.compute_tier.Tier (row)
            trial tier computer
        """
        raise NotImplementedError()


# -- Tier Types --


@schema.lookup
class RandomSplit(_Tier):
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
    -> TrialSet
    -> Trial
    -> Tier
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
        tiers = (Tier & key).link.tier.tiers(trials=trials)

        for index, trials in enumerate(tiers):

            # trial tier keys
            keys = [dict(key, tier_index=index, **trial) for trial in trials.fetch("KEY")]

            # insert
            self.insert(keys)
