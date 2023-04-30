import numpy as np
import datajoint as dj
from djutils import merge, skip_missing
from foundation.scan import trial as scan_trial, unit as scan_unit
from foundation.recording import trial, trace
from foundation.schemas import recording as schema


@schema
class ScanTrials(dj.Computed):
    definition = """
    -> scan_trial.FilteredTrials.proj(scan_trial_filters_id='trial_filters_id')
    -> trial.TrialFilterSet
    ---
    -> trial.TrialSet
    """

    @skip_missing
    def make(self, key):
        # filtered scan trials
        trials = scan_trial.FilteredTrials.proj(..., scan_trial_filters_id="trial_filters_id") & key
        trials = scan_trial.TrialSet & trials
        trials = merge(trials.members, trial.TrialLink.ScanTrial)
        trials = trial.TrialLink & trials

        # filter trials
        for filter_key in (trial.TrialFilterSet & key).members.fetch(dj.key, order_by="member_id"):
            trials = (trial.TrialFilterLink & key).link.filter(trials)

        # trial set
        trials = trial.TrialSet.fill(trials, prompt=False)
        self.insert1(dict(key, **trials))
