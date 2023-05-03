import os
import numpy as np
from djutils import Files
from foundation.stimulus import video
from foundation.utility import resize
from foundation.schemas import stimulus as schema


@schema.computed
class ResizedVideo(Files):
    store = "scratch09"
    definition = """
    -> video.VideoLink
    -> resize.ResizeLink
    -> resize.Resolution
    ---
    video       : filepath@scratch09    # npy file, [frames, height, width, channels]
    """

    def make(self, key):
        # video, resize links and resolution
        video_link = video.VideoLink & key
        resize_link = resize.ResizeLink & key
        height, width = (resize.Resolution & key).fetch1("height", "width")

        # resize video
        vid = video_link.resized_video(resize_link=resize_link, height=height, width=width)

        # save video
        file = os.path.join(self.tuple_dir(key, create=True), "video.npy")
        np.save(file, vid.array)

        # insert key
        self.insert1(dict(key, video=file))
