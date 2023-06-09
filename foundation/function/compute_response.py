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
        from foundation.recording.compute_trace import VisualTrace

        return (VisualTrace & self.key & {"video_id": video_id}).response

    @rowproperty
    def timing(self):
        from foundation.utility.resample import Rate, Offset

        return (Rate & self.key).link.period, (Offset & self.key).link.offset


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
        from foundation.utils.response import Trials
        from foundation.fnn.compute_output import NetworkOutput

        # fetch attributes
        filterset, index, perspective, modulation = self.key.fetch1(
            "trial_filterset_id", "response_index", "perspective", "modulation"
        )

        # compute output
        output = NetworkOutput & self.key
        output = output.visual_recording(
            video_id=video_id,
            trial_filterset_id=filterset,
            perspective=perspective,
            modulation=modulation,
        )

        # response trials
        return Trials(data=[o[:, index] for o in output.values], index=output.index)

    @rowproperty
    def timing(self):
        from foundation.fnn.network import Network

        return (Network & self.key).link.data.link.timing
