import numpy as np
import pandas as pd
from djutils import keys, merge, rowmethod, rowproperty
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
        foundation.utils.response.Trials
            response trials
        """
        raise NotImplementedError()

    @rowproperty
    def timing(self):
        """
        Returns
        -------
        float
            sampling period (seconds)
        float
            response offset (seconds)
        """
        raise NotImplementedError()


# -- Response Types --


@keys
class TrialResponse(Response):
    """Recording Trial Response"""

    @property
    def key_list(self):
        return [
            function.TrialResponse,
        ]

    @rowmethod
    def visual(self, video_id):
        from foundation.recording.compute_visual import Trace
        from foundation.utils.response import Trials

        # visual responses
        responses, trial_ids = (Trace & self.key & {"video_id": video_id}).responses

        # response trials
        return Trials(data=responses.T, index=trial_ids)

    @rowproperty
    def timing(self):
        from foundation.utility.resample import Rate, Offset

        # sampling period
        period = (Rate & self.key).link.period

        # response offset
        offset = (Offset & self.key).link.offset

        # response timing
        return period, offset


@keys
class FnnTrialResponse(Response):
    """Fnn Recording Trial Response"""

    @property
    def key_list(self):
        return [
            function.FnnTrialResponse,
        ]

    @rowmethod
    def visual(self, video_id):
        from foundation.fnn.compute_visual import NetworkModelTrial
        from foundation.utils.response import Trials

        # visual responses
        responses, trial_ids = (NetworkModelTrial & self.key & {"video_id": video_id}).responses

        # response index
        index = self.key.fetch1("response_index")

        # response trials
        if trial_ids is None:
            return Trials(data=[responses[:, index]], index=None)
        else:
            return Trials(data=responses[:, :, index].T, index=trial_ids)

    @rowproperty
    def timing(self):
        from foundation.fnn.network import Network

        # response timing
        return (Network & self.key).link.data.timing
