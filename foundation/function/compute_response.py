import numpy as np
import pandas as pd
from djutils import keys, merge, rowproperty
from foundation.virtual import utility, stimulus, recording, function


# ----------------------------- Response -----------------------------

# -- Response Base --


class Response:
    """Functional Response"""

    @rowproperty
    def response(self):
        """
        Returns
        -------
        pandas.Series
            index -- str | None
                : unique trial identifier | None
            data -- 1D array
                : [timepoints] ; response trace
        """
        raise NotImplementedError()


# -- Visual Response Types --


class VisualRecording(Visual):
    """Visual Recording Response"""

    @property
    def key_list(self):
        return [
            stimulus.Video,
            function.Recording,
        ]

    @rowproperty
    def response(self):
        from foundation.recording.trial import Trial, TrialVideo, TrialSet, TrialFilterSet
        from foundation.recording.compute_trace import ResampledTrace

        # all trials
        trials = merge(self.key, recording.TraceTrials)
        trials = Trial & (TrialSet & trials).members

        # restricted trials
        trials = (TrialFilterSet & self.key).filter(trials)
        trials = merge(trials, TrialVideo) & self.key

        # trial response
        return (ResampledTrace & trials & self.key).trials
