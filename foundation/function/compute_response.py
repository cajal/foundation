import numpy as np
import pandas as pd
from djutils import keys, merge, rowmethod
from foundation.virtual import utility, stimulus, recording, function


# ----------------------------- Response -----------------------------

# -- Response Base --


class Response:
    """Functional Response"""

    @rowmethod
    def visual(self, video_id):
        """
        Parameters
        ----------
        video_id : str
            key (foundation.stimulus.video.Video)

        Returns
        -------
        foundation.utils.response.Response
            response trials
        """
        raise NotImplementedError()


# -- Visual Response Types --


@keys
class Recording(Response):
    """Recording Response"""

    @property
    def key_list(self):
        return [
            function.Recording,
        ]

    @rowmethod
    def visual(self, video_id):
        """
        Parameters
        ----------
        video_id : str
            key (foundation.stimulus.video.Video)

        Returns
        -------
        pandas.Series
            index -- str | None
                : trial_id -- key (foundation.recording.trial.Trial) | None
            data -- 1D array
                : [samples] ; response trace
        """
        from foundation.utils.response import Response
        from foundation.recording.trial import Trial, TrialVideo, TrialSet, TrialFilterSet
        from foundation.recording.compute_trace import ResampledTrace

        # all trials
        trials = merge(self.key, recording.TraceTrials)
        trials = Trial & (TrialSet & trials).members

        # restricted trials
        trials = (TrialFilterSet & self.key).filter(trials)
        trials = merge(trials, TrialVideo) & self.key & {"video_id": video_id}

        # response trials
        trials = (ResampledTrace & trials & self.key).trials

        return Response(data=trials.tolist(), index=trials.index)
