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
        foundation.utils.response.Response
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
class Recording(Response):
    """Recording Response"""

    @property
    def key_list(self):
        return [
            function.Recording,
        ]

    @rowmethod
    def visual(self, video_id):
        from foundation.recording.compute_trace import Visual

        return (Visual & self.key & {"video_id": video_id}).response

    @rowproperty
    def timing(self):
        from foundation.utility.resample import Rate, Offset

        return (Rate & self.key).link.period, (Offset & self.key).link.offset


@keys
class FnnRecording(Response):
    """Fnn Recording Response"""

    @property
    def key_list(self):
        return [
            function.FnnRecording,
        ]

    @rowmethod
    def visual(self, video_id):
        from foundation.utils.response import Response
        from foundation.fnn.compute_output import NetworkOutput

        # fetch attributes
        filterset, perpectives, modulations, index = self.key.fetch1(
            "trial_filterset_id", "trial_perspectives", "trial_modulations", "response_index"
        )

        # compute output
        output = NetworkOutput & self.key
        output = output.visual_recording(
            video_id=video_id,
            trial_filterset_id=filterset,
            trial_perspectives=perpectives,
            trial_modulations=modulations,
        )

        return Response(data=[o[:, index] for o in output.values], index=output.index)

    @rowproperty
    def timing(self):
        from foundation.fnn.network import Network

        return (Network & self.key).link.data.link.timing
