import numpy as np
import datajoint as dj
from djutils import link, group, merge, row_method, skip_missing
from foundation.scan import timing, pupil
from foundation.schemas.pipeline import pipe_stim, pipe_shared
from foundation.schemas import scan as schema


# -------------- Trial Set --------------


@group(schema)
class TrialSet:
    keys = [pipe_stim.Trial]
    name = "scan_trials"
    comment = "set of scan trials"


# -------------- Trial Filter --------------

# -- Trial Filter Base --


class TrialFilterBase:
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


# -- Trial Restriction Types --


@schema
class PupilNansFilter(dj.Lookup):
    definition = """
    -> pipe_shared.TrackingMethod
    max_nans        : decimal(4, 3)     # maximum tolerated fraction of nans
    """

    @row_method
    def filter(self, trials):
        max_nans = self.fetch1("max_nans")
        key = merge(trials, pupil.PupilNans) & f"nans < {max_nans}"
        return trials & key.proj()


# -- Trial Filter Link --


@link(schema)
class TrialFilterLink:
    links = [PupilNansFilter]
    name = "scan_trial_filter"
    comment = "scan trial filter"


@group(schema)
class TrialFilterSet:
    keys = [TrialFilterLink]
    name = "scan_trial_filters"
    comment = "set of scan trial filters"


# -- Computed Trial Filter --


@schema
class FilteredTrials(dj.Computed):
    definition = """
    -> timing.Timing
    -> TrialFilterSet
    ---
    -> TrialSet
    """

    @skip_missing
    def make(self, key):
        # scan trials
        trials = pipe_stim.Trial & key

        # filter trials
        for filter_key in (TrialFilterSet & key).members.fetch(dj.key, order_by="member_id"):
            trials = (TrialFilterLink & key).link.filter(trials)

        # insert trial set
        trial_set = TrialSet.fill(trials, prompt=False)

        # insert key
        self.insert1(dict(key, **trial_set))
