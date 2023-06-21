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
class RecordingResponse(_Response):
    definition = """
    -> recording.Trace              # response trace
    -> recording.TrialFilterSet     # trial filter
    -> recording.TrialSet           # standardization trial set
    -> utility.Standardize          # standardization method
    -> utility.Resample             # resampling method
    -> utility.Offset               # resampling offset
    -> utility.Rate                 # resampling rate
    """

    @rowproperty
    def response(self):
        from foundation.function.compute_response import RecordingResponse

        return RecordingResponse & self


@schema.lookup
class FnnRecordingResponse(_Response):
    definition = """
    -> fnn.NetworkModel                 # network model
    -> recording.TrialFilterSet         # trial filter
    trial_perspectives  : bool          # trial perspectives
    trial_modulations   : bool          # trial modulations
    response_index      : int unsigned  # network response index
    """

    @rowproperty
    def response(self):
        from foundation.function.compute_response import FnnRecordingResponse

        return FnnRecordingResponse & self


# -- Response --


@schema.link
class Response:
    links = [RecordingResponse, FnnRecordingResponse]
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
    -> Response
    -> stimulus.VideoSet
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
    -> ResponseSet
    -> stimulus.VideoSet
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
        videos = tqdm(videos, desc="Videos")

        # response pair
        key_x, key_y = (ResponseSet & key).members.fetch("response_id", order_by="response_id", as_dict=True)
        response_x = (Response & key_x).link.response
        response_y = (Response & key_y).link.response
        x = []
        y = []

        # compute response set
        with disable_tqdm():
            for video in videos:

                # video responses
                _x = response_x.visual(video_id=video)
                _y = response_y.visual(video_id=video)

                # verify response pair
                assert _x.matches(_y)

                # append
                x.append(_x)
                y.append(_y)

        # concatenate responses
        x = concatenate(*x, burnin=key["burnin"])
        y = concatenate(*y, burnin=key["burnin"])

        # response correlation
        correlation = (Correlation & key).link.correlation(x, y)

        # insert
        self.insert1(dict(key, correlation=correlation))
