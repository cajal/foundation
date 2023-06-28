from djutils import rowproperty
from foundation.virtual import utility, stimulus, recording, fnn
from foundation.schemas import function as schema


# ----------------------------- Direction -----------------------------


# -- Direction Interface --


class DirectionType:
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


# @schema.lookup
# class RecordingDirection(DirectionType):
#     definition = """
#     -> recording.Trace              # response trace
#     -> recording.TrialFilterSet     # trial filter
#     """


# # -- Direction --


# @schema.link
# class Direction:
#     links = [RecordingDirection]
#     name = "direction"
#     comment = "direction tuning"


# # -- Computed Direction --


# @schema.computed
# class VisualDirectionImpulse:
#     definition = """
#     -> Direction
#     -> stimulus.VideoSet
#     -> utility.Impulse
#     ---
#     degrees         : longblob      # [directions] -- degrees from 0 to 360
#     activations     : longblob      # [directions] -- activations
#     """

#     def make(self, key):
#         pass
