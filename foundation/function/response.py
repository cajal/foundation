from djutils import rowproperty
from foundation.utils import tqdm, disable_tqdm
from foundation.virtual import utility, stimulus, recording, fnn
from foundation.schemas import function as schema


# ----------------------------- Response -----------------------------

# -- Response Base --


class _Response:
    """Functional Response"""

    @rowproperty
    def response(self):
        """
        Returns
        -------
        foundation.function.compute_response.Response (row)
            functional response
        """
        raise NotImplementedError()


# -- Response Types --


@schema.lookup
class Recording(_Response):
    definition = """
    -> recording.Trace
    -> recording.TrialFilterSet
    -> utility.Resample
    -> utility.Offset
    -> utility.Rate
    """

    @rowproperty
    def response(self):
        from foundation.function.compute_response import Recording

        return Recording & self


@schema.lookup
class FnnRecording(_Response):
    definition = """
    -> fnn.ModelNetwork
    -> recording.TrialFilterSet
    trial_perspectives  : bool          # recording trial perspectives
    trial_modulations   : bool          # recording trial modulations
    response_index      : int unsigned  # response index
    """

    @rowproperty
    def response(self):
        from foundation.function.compute_response import FnnRecording

        return FnnRecording & self


# -- Response --


@schema.link
class Response:
    links = [Recording, FnnRecording]
    name = "response"
    comment = "functional response"


# -- Computed Response --


@schema.computed
class VisualResponseMeasure:
    definition = """
    -> stimulus.VideoSet
    -> Response
    -> utility.Measure
    ---
    measure = NULL      : float     # visual response measure
    """

    def make(self, key):
        from foundation.utils.response import concatenate
        from foundation.utility.measure import Measure
        from foundation.stimulus.video import VideoSet

        # videos
        videos = (VideoSet & key).members.fetch("video_id", order_by="video_id")
        videos = tqdm(videos, desc="Videos")

        # video responses
        with disable_tqdm():
            response = (Response & key).link.response
            response = concatenate([response.visual(video_id=v) for v in videos])

        # response measure
        measure = (Measure & key).link.measure(response)

        # insert
        self.insert1(dict(key, measure=measure))
