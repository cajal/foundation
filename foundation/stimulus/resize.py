import numpy as np
from foundation.virtual import utility
from foundation.stimulus.video import Video
from foundation.schemas import stimulus as schema


@schema.computed
class ResizedVideo:
    definition = """
    -> Video
    -> utility.Resize
    -> utility.Resolution
    ---
    video           : blob@external    # [frames, height, width, channels]
    """

    def make(self, key):
        from foundation.stimulus.compute.resize import ResizedVideo

        # resize video
        video = (ResizedVideo & key).video

        # insert key
        self.insert1(dict(key, video=video.array))
