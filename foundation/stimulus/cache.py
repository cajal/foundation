import numpy as np
from djutils import Filepath
from foundation.virtual import utility
from foundation.stimulus.video import VideoLink
from foundation.schemas import stimulus as schema


@schema.computed
class ResizedVideo(Filepath):
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
        filepath = self.createpath(key, "video", "npy")
        np.save(filepath, video.array)

        # insert key
        self.insert1(dict(key, video=filepath))
