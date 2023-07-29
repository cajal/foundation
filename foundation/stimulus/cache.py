import numpy as np
from foundation.virtual import utility
from foundation.stimulus.video import Video
from foundation.schemas import stimulus as schema


@schema.computed
class ResizedVideoTemp:
    definition = """
    -> Video
    -> utility.Resize
    -> utility.Resolution
    ---
    video       : blob@external    # npy file, [frames, height, width, channels]
    """

    @property
    def key_source(self):
        return ResizedVideo.proj()

    def make(self, key):
        v = (ResizedVideo & key).fetch1("video")
        self.insert1(dict(key, video=np.load(v)))


@schema.computed
class ResizedVideo:
    definition = """
    -> Video
    -> utility.Resize
    -> utility.Resolution
    ---
    video           : blob@external    # npy file, [frames, height, width, channels]
    """

    @property
    def key_source(self):  # TEMP
        return ResizedVideoTemp.proj()

    def make(self, key):  # TEMP
        v = (ResizedVideoTemp & key).fetch1("video")
        self.insert1(dict(key, video=v))

    # def make(self, key):
    #     from foundation.stimulus.compute_video import ResizedVideo

    #     # resize video
    #     video = (ResizedVideo & key).video

    #     # save video
    #     filepath = self.createpath(key, "video", "npy")
    #     np.save(filepath, video.array)

    #     # insert key
    #     self.insert1(dict(key, video=filepath))
