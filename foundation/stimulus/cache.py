import os
import numpy as np
from djutils import Files
from foundation.stimulus import video
from foundation.utility import resize
from foundation.schemas import stimulus as schema


@schema.computed
class CachedVideo(Files):
    store = "scratch09"
    definition = """
    -> video.VideoLink
    -> resize.ResizeLink
    -> resize.Resolution
    ---
    video       : filepath@scratch09    # npy file, [frames, height, width, channels]
    """

    def make(self, key):
        # load video
        vid = (video.VideoLink & key).link.video

        # target resolution
        height, width = (resize.Resolution & key).fetch1("height", "width")

        # resize video
        vid = (resize.ResizeLink & key).link.resize(video=vid, height=height, width=width)

        # save video
        file = os.path.join(self.tuple_dir(key, create=True), "video.npy")
        np.save(file, vid.array)

        # insert key
        self.insert1(dict(key, video=file))
