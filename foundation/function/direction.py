from djutils import rowproperty
from foundation.virtual import utility, stimulus, recording, fnn
from foundation.schemas import function as schema


# ----------------------------- Direction -----------------------------


# @schema.lookup
# class DirectionPrecision:
#     definition = """
#     precision   :
#     """


# -- Direction Base --


class _Direction:
    """Direction Tuning"""

    @rowproperty
    def direction(self):
        """
        Returns
        -------
        foundation.function.compute_direction.Direction (row)
            direction tuning
        """
        raise NotImplementedError()


# -- Direction Types --


@schema.lookup
class RecordingDirection(_Direction):
    definition = """
    -> recording.Trace              # response trace
    -> recording.TrialFilterSet     # trial filter
    """


# -- Direction --


@schema.link
class Direction:
    links = [RecordingDirection]
    name = "direction"
    comment = "direction tuning"
