from djutils import merge, rowproperty, rowmethod
from foundation.virtual import recording, fnn
from foundation.schemas import tuning as schema


# ---------------------------- DirectionResp ----------------------------

# -- DirectionResp Interface --


class DirectionResp:
    """Unit responses to direction stimuli"""

    @rowproperty
    def compute(self):
        """
        Returns
        -------
        foundation.tuning.compute.direction (row)
            compute unit responses to direction stimuli
        """
        raise NotImplementedError()
    

# -- DirectionResp Types --
    

@schema.lookup
class RecordingDirectionResp(DirectionResp):
    definition = """
    -> pipe_shared.Recording
    """

    @rowproperty
    def compute(self):
        from foundation.tuning.compute.direction import RecordingDirectionResp

        return RecordingDirectionResp & self
    

@schema.lookup
class FnnDirectionResp(DirectionResp):
    definition = """
    -> fnn.Model
    """

    @rowproperty
    def compute(self):
        from foundation.tuning.compute.direction import FnnDirectionResp

        return FnnDirectionResp & self
    

# -- DirectionResp --
    

@schema.link
class DirectionResp:
    links = [RecordingDirectionResp, FnnDirectionResp]
    name = "direction_resp"
    comment = "unit responses to direction stimuli"


# -- OSI --
    
@schema.computed
class OSI:
    definition = """
    -> DirectionResp
    -> recording.Trace
    ---
    osi = NULL      : float     # orientation selectivity index
    gosi = NULL     : float     # global orientation selectivity index
    """


# -- DSI --

@schema.computed
class DSI:
    definition = """
    -> DirectionResp
    -> recording.Trace
    ---
    dsi = NULL      : float     # direction selectivity index
    gdsi = NULL     : float     # global direction selectivity index
    """
