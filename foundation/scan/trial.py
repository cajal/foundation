from djutils import merge, rowmethod
from foundation.scan.experiment import Scan
from foundation.scan.pupil import PupilNans
from foundation.virtual.bridge import pipe_stim, pipe_shared
from foundation.schemas import scan as schema


# -------------- Trial Set --------------


@schema.set
class TrialSet:
    keys = [pipe_stim.Trial]
    name = "trials"
    comment = "scan trial set"
    part_name = "trial"


# -------------- Trial Filter --------------

# -- Filter Types --


@schema.filter_lookup
class PupilNansFilter:
    ftype = pipe_stim.Trial
    definition = """
    -> pipe_shared.TrackingMethod
    max_nans        : decimal(4, 3)     # maximum tolerated fraction of nans
    """

    @rowmethod
    def filter(self, trials):
        key = merge(trials, self, PupilNans)

        return trials & (key & "nans < max_nans").proj()


# -- Filter --


@schema.filter_link
class TrialFilterLink:
    links = [PupilNansFilter]
    name = "trial_filter"
    comment = "scan trial filter"


# -- Filter Set --


@schema.filter_link_set
class TrialFilterSet:
    link = TrialFilterLink
    name = "trial_filters"
    comment = "scan trial filter set"


# -- Computed Filter --


@schema.computed
class FilteredTrials:
    definition = """
    -> Scan
    -> TrialFilterSet
    ---
    -> TrialSet
    """

    def make(self, key):
        # scan trials
        trials = pipe_stim.Trial & key

        # filter trials
        trials = (TrialFilterLink & key).filter(trials)

        # insert trial set
        trial_set = TrialSet.fill(trials, prompt=False)

        # insert key
        self.insert1(dict(key, **trial_set))
