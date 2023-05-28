import numpy as np
import pandas as pd
from djutils import keys, merge, rowproperty
from foundation.virtual import utility, stimulus, recording, function


# ----------------------------- Visual Response -----------------------------

# -- Visual Response Base --


@keys
class Visual:
    """Visual Response"""

    @property
    def key_list(self):
        return [stimulus.Video] + self.response_list

    @property
    def response_list(self):
        raise NotImplementedError()

    @rowproperty
    def response(self):
        """
        Returns
        -------
        pandas.Series
            index -- str : unique trial identifier
            data -- 1D array : visual response
        """
        raise NotImplementedError()


# -- Visual Response Types --


class VisualRecording(Visual):
    """Visual Recording Response"""

    @property
    def response_list(self):
        return [function.Recording]

    @rowproperty
    def response(self):
        from foundation.recording.trial import Trial, TrialSet, TrialFilterSet
        from foundation.recording.compute_trace import ResampledTrace

        # all trials
        trials = merge(self.key, recording.TraceTrials)
        trials = Trial & (TrialSet & trials).members

        # restricted trials
        trials = (TrialFilterSet & self.key).filter(trials)
        trials = merge(trials, recording.TrialVideo) & self.key

        # trial response
        return (ResampledTrace & trials & self.key).trials
