import os
import numpy as np
from djutils import Files
from foundation.virtual import utility
from foundation.stimulus.video import VideoLink
from foundation.schemas import stimulus as schema


@schema.computed
class ResizedVideo(Files):
    store = "scratch09"
    definition = """
    -> VideoLink
    -> utility.ResizeLink
    -> utility.Resolution
    ---
    video       : filepath@scratch09    # npy file, [frames, height, width, channels]
    """

    def make(self, key):
        from foundation.stimulus.compute import ResizeVideo

        # resize video
        video = (ResizeVideo & key).video

        # save video
        file = os.path.join(self.tuple_dir(key, create=True), "video.npy")
        np.save(file, video.array)

        # insert key
        self.insert1(dict(key, video=file))
