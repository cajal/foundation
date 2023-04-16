import numpy as np
import datajoint as dj
from PIL import Image
from foundation.utils.logging import logger

stimulus = dj.create_virtual_module("stimulus", "pipeline_stimulus")


schema = dj.schema("foundation_stimuli")


class ConditionMixin:
    @property
    def frames(self):
        """
        Returns
        -------
        List[Image]
            stimulus frames
        """
        raise NotImplementedError


@schema
class Clip(dj.Computed, ConditionMixin):
    definition = """
    -> stimulus.Clip
    ---
    frames      : int unsigned  # number of stimulus frames
    """
