from djutils import merge, rowmethod
from foundation.scan.experiment import Scan
from foundation.scan.pupil import PupilNans
from foundation.virtual.bridge import pipe_stim, pipe_shared
from foundation.schemas import scan as schema


# -------------- Trial Set --------------


@schema.set
class TrialSet:
    keys = [pipe_stim.Trial]
    name = "trialset"
    comment = "scan trial set"


# -------------- Trial Filter --------------

# -- Filter Types --


@schema.lookupfilter
class PupilNansFilter:
    filtertype = pipe_stim.Trial
    definition = """
    -> pipe_shared.TrackingMethod
    max_nans        : decimal(4, 3)     # maximum tolerated fraction of nans
    """

    @rowmethod
    def filter(self, trials):
        key = merge(trials, self, PupilNans)

        return trials & (key & "nans < max_nans").proj()


# -- Filter --


@schema.filterlink
class TrialFilter:
    links = [PupilNansFilter]
    name = "trial_filter"
    comment = "scan trial filter"


# -- Filter Set --


@schema.filterlinkset
class TrialFilterSet:
    link = TrialFilter
    name = "trial_filterset"
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
        trials = (TrialFilter & key).filter(trials)

        # insert trial set
        trial_set = TrialSet.fill(trials, prompt=False)

        # insert key
        self.insert1(dict(key, **trial_set))
