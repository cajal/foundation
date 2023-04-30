import numpy as np
import datajoint as dj
from djutils import merge, skip_missing
from foundation.scan import trial as scan_trial, unit as scan_unit
from foundation.recording import trial, trace
from foundation.schemas import recording as schema


@schema
class ScanTrials(dj.Computed):
    definition = """
    -> scan_trial.FilteredTrials
    ---
    -> trial.TrialSet
    """

    @skip_missing
    def make(self, key):
        # filtered scan trials
        trials = scan_trial.TrialSet & (scan_trial.FilteredTrials & key)
        trials = merge(trials.members, trial.TrialLink.ScanTrial)

        # trial set
        trials = trial.TrialSet.fill(trials, prompt=False)
        self.insert1(dict(key, **trials))
