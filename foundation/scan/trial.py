from djutils import merge, row_method
from foundation.scan import timing, pupil
from foundation.schemas.pipeline import pipe_stim, pipe_shared
from foundation.schemas import scan as schema


# -------------- Trial Set --------------


@schema.set
class TrialSet:
    keys = [pipe_stim.Trial]
    name = "trials"
    comment = "set of scan trials"


# -------------- Trial Filter --------------

# -- Trial Filter Base --


class _TrialFilter:
    """Trial Filter"""

    @row_method
    def filter(self, trials):
        """
        Parameters
        ----------
        trials : pipe_stim.Trial
            tuples from pipe_stim.Trial

        Returns
        -------
        pipe_stim.Trial
            restricted tuples
        """
        raise NotImplementedError()


# -- Trial Filter Types --


@schema.lookup
class PupilNansFilter(_TrialFilter):
    definition = """
    -> pipe_shared.TrackingMethod
    max_nans        : decimal(4, 3)     # maximum tolerated fraction of nans
    """

    @row_method
    def filter(self, trials):
        key = merge(trials, self, pupil.PupilNans)

        return trials & (key & "nans < max_nans").proj()


# -- Trial Filter Link --


@schema.link
class TrialFilterLink:
    links = [PupilNansFilter]
    name = "trial_filter"
    comment = "scan trial filter"


@schema.set
class TrialFilterSet:
    keys = [TrialFilterLink]
    name = "trial_filters"
    comment = "set of scan trial filters"


# -- Computed Trial Filter --


@schema.computed
class FilteredTrials:
    definition = """
    -> timing.Timing
    -> TrialFilterSet
    ---
    -> TrialSet
    """

    def make(self, key):
        # scan trials
        trials = pipe_stim.Trial & key

        # filter trials
        for filter_key in (TrialFilterSet & key).members.fetch("KEY", order_by="member_id"):
            trials = (TrialFilterLink & key).link.filter(trials)

        # insert trial set
        trial_set = TrialSet.fill(trials, prompt=False)

        # insert key
        self.insert1(dict(key, **trial_set))
