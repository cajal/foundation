from djutils import rowproperty
from foundation.utils import tqdm, disable_tqdm
from foundation.virtual import utility, stimulus, recording, fnn
from foundation.schemas import function as schema


# ----------------------------- Response -----------------------------

# -- Response Base --


class _Response:
    """Functional Response"""

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

    @rowproperty
    def response(self):
        """
        Returns
        -------
        foundation.function.response.Response
            functional response
        """
        raise NotImplementedError()


# -- Response Types --


@schema.lookup
class Recording(_Response):
    definition = """
    -> recording.TrialFilterSet
    -> recording.Trace
    -> utility.Resample
    -> utility.Offset
    -> utility.Rate
    """

    @rowproperty
    def timing(self):
        from foundation.utility.resample import Rate, Offset

        return (Rate & self).link.period, (Offset & self).link.offset

    @rowproperty
    def response(self):
        from foundation.function.compute_response import Recording

        return Recording & self


@schema.lookup
class FnnRecording(_Response):
    definition = """
    -> recording.TrialFilterSet
    -> fnn.ModelNetwork
    response_index      : int unsigned  # response index
    """

    # @rowproperty
    # def timing(self):
    #     from foundation.fnn.dataspec import DataSp


# -- Response --


@schema.link
class Response:
    links = [Recording]
    name = "response"
    comment = "functional response"


# -- Computed Response --


@schema.computed
class VisualMeasure:
    definition = """
    -> stimulus.VideoSet
    -> Response
    """

    def make(self, key):
        pass
