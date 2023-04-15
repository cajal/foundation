import datajoint as dj
from djutils import link


schema = dj.schema("foundation_stimuli")


@schema
class FrameMode(dj.Lookup):
    definition = """
    frame_mode              : varchar(8)    # stimulus frame mode
    ---
    frame_mode_description  : varchar(1024) # description of stimulus frame mode
    """


@schema
class RateMode(dj.Lookup):
    definition = """
    rate_mode               : varchar(8)    # stimulus frame rate mode
    ---
    rate_mode_description   : varchar(1024) # description of stimulus frame rate mode
    """
