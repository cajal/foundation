from djutils import rowproperty, cache_rowproperty
from foundation.utils import tqdm, disable_tqdm
from foundation.virtual import utility, stimulus, recording, fnn
from foundation.schemas import function as schema


# ----------------------------- Response -----------------------------

# -- Response Interface --


class ResponseType:
    """Functional Response"""

    @rowproperty
    def compute(self):
        """
        Returns
        -------
        foundation.function.compute_response.Response (row)
            functional response
        """
        raise NotImplementedError()


# -- Response Types --


@schema.lookup
class TrialResponse(ResponseType):
    definition = """
    -> recording.Trace                  # trace
    -> recording.TrialFilterSet         # trial filter
    -> recording.TrialSet               # standardization trial set
    -> utility.Standardize              # standardization method
    -> utility.Resample                 # resampling method
    -> utility.Offset                   # resampling offset
    -> utility.Rate                     # resampling rate
    """

    @rowproperty
    def compute(self):
        from foundation.function.compute_response import RecordingResponse

        return RecordingResponse & self


@schema.lookup
class FnnTrialResponse(ResponseType):
    definition = """
    -> fnn.NetworkModel                 # network model
    -> recording.TrialFilterSet         # trial filter
    trial_perspective   : bool          # trial | default perspective
    trial_modulation    : bool          # trial | default modulation
    response_index      : int unsigned  # response index
    """

    @rowproperty
    def compute(self):
        from foundation.function.compute_response import FnnRecordingResponse

        return FnnRecordingResponse & self


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
            compute = (Response & key).link.compute
            response = (compute.visual(video_id=v) for v in videos)
            response = concatenate(*response, burnin=key["burnin"])

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
        compute_x = (Response & key_x).link.compute
        compute_y = (Response & key_y).link.compute
        x = []
        y = []

        # compute response set
        with disable_tqdm():
            for video in videos:

                # video responses
                _x = compute_x.visual(video_id=video)
                _y = compute_y.visual(video_id=video)

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
