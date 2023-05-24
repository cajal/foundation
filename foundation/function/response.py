from djutils import rowproperty
from foundation.virtual import utility, recording, fnn
from foundation.schemas import function as schema


# -------------- Response --------------

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
            sampling offset (seconds)
        """
        raise NotImplementedError()


# -- Response Types --


@schema.lookup
class Recording(_Response):
    definition = """
    -> recording.Trace
    -> utility.Resample
    -> utility.Offset
    -> utility.Rate
    """

    @rowproperty
    def timing(self):
        from foundation.utility.resample import Rate, Offset

        return (Rate & self).link.period, (Offset & self).link.offset


@schema.lookup
class Fnn(_Response):
    definition = """
    -> fnn.ModelNetwork
    response_index      : int unsigned  # response index
    """

    # @rowproperty
    # def timing(self):
    #     from foundation.fnn.dataspec import DataSp



# -- Response --


@schema.link
class Response:
    links = [Recording, Fnn]
    name = "response"
    comment = "functional response"
