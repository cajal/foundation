import numpy as np
import datajoint as dj
from djutils import link, row_property, MissingError, RestrictionError
from foundation.utils.logging import logger
from foundation.utils.trace import truncate
from foundation.bridges.pipeline import pipe_exp
from foundation.recordings import trial

schema = dj.schema("foundation_recording")


# -------------- Recording --------------

# -- Recording Base --


class RecordingBase:
    """Recording"""

    @row_property
    def times(self):
        """
        Returns
        -------
        times : 1D array
            recording times
        """
        raise NotImplementedError()

    @row_property
    def trial_flips(self):
        """
        Returns
        -------
        trials.TrialFlips
            tuples from trials.TrialFlips
        """
        raise NotImplementedError()


# -- Trace Types --


class ScanBase(RecordingBase):
    """Scan Recording"""

    @row_property
    def scan_key(self):
        key = ["animal_id", "session", "scan_idx"]
        return dict(zip(key, self.fetch1(*key)))

    @row_property
    def trial_flips(self):
        from foundation.bridges.pipeline import pipe_stim

        scan_trials = pipe_stim.Trial.proj() & self
        keys = trial.TrialFlips.proj() * trial.TrialLink.ScanTrial * scan_trials

        if scan_trials - keys:
            raise MissingError("Missing trials.")

        if keys - scan_trials:
            raise RestrictionError("Unexpected trials.")

        return trial.TrialFlips & keys
