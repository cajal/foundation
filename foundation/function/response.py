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
class TrialResponse(_Response):
    definition = """
    -> recording.Trace
    -> recording.TrialFilterSet
    -> recording.TrialSet
    -> utility.Standardize
    -> utility.Resample
    -> utility.Offset
    -> utility.Rate
    """

    @rowproperty
    def response(self):
        from foundation.function.compute_response import TrialResponse

        return TrialResponse & self


@schema.lookup
class FnnTrialResponse(_Response):
    definition = """
    -> fnn.ModelNetwork
    -> recording.TrialFilterSet
    response_index              : int unsigned  # response index
    perspective                 : bool          # use recording perspective
    modulation                  : bool          # use recording modulation
    """

    @rowproperty
    def response(self):
        from foundation.function.compute_response import FnnTrialResponse

        return FnnTrialResponse & self


# -- Response --


@schema.link
class Response:
    links = [TrialResponse, FnnTrialResponse]
    name = "response"
    comment = "functional response"


@schema.linkset
class ResponseSet:
    link = Response
    name = "responseset"
    comment = "functional response set"


# -- Computed Response --


@schema.computed
class VisualResponseMeasure:
    definition = """
    -> stimulus.VideoSet
    -> Response
    -> utility.Measure
    -> utility.Burnin
    ---
    measure = NULL          : float     # visual response measure
    """

    def make(self, key):
        from foundation.utils.response import concatenate
        from foundation.utility.response import Measure
        from foundation.stimulus.video import VideoSet

        # videos
        videos = (VideoSet & key).members.fetch("video_id", order_by="video_id")
        videos = tqdm(videos, desc="Videos")

        # video responses
        with disable_tqdm():
            response = (Response & key).link.response
            response = concatenate(
                *(response.visual(video_id=v) for v in videos),
                burnin=key["burnin"],
            )

        # response measure
        measure = (Measure & key).link.measure(response)

        # insert
        self.insert1(dict(key, measure=measure))


@schema.computed
class VisualResponseCorrelation:
    definition = """
    -> stimulus.VideoSet
    -> ResponseSet
    -> utility.Correlation
    -> utility.Burnin
    ---
    correlation = NULL      : float     # visual response correlation
    """

    @property
    def key_source(self):
        return (
            stimulus.VideoSet.proj()
            * (ResponseSet & "members=2").proj()
            * utility.Correlation.proj()
            * utility.Burnin.proj()
        )

    def make(self, key):
        from foundation.utils.response import concatenate
        from foundation.utility.response import Correlation
        from foundation.stimulus.video import VideoSet

        # videos
        videos = (VideoSet & key).members.fetch("video_id", order_by="video_id")
        videos_x = tqdm(videos, desc="Videos -- X")
        videos_y = tqdm(videos, desc="Videos -- Y")

        # responses
        responses = (ResponseSet & key).members
        responses = responses.fetch("response_id", order_by="response_id", as_dict=True)

        # video responses
        xy = []
        with disable_tqdm():
            for videos, response in zip([videos_x, videos_y], responses):
                response = (Response & response).link.response
                response = concatenate(
                    *(response.visual(video_id=v) for v in videos),
                    burnin=key["burnin"],
                )
                xy.append(response)

        # response correlation
        correlation = (Correlation & key).link.correlation(*xy)

        # insert
        self.insert1(dict(key, correlation=correlation))
