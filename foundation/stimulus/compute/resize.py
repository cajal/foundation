from djutils import keys, rowproperty
from foundation.virtual import utility, stimulus


# ---------------------------- Resize ----------------------------


@keys
class ResizedVideo:
    """Resized Video"""

    @property
    def keys(self):
        return [
            stimulus.Video,
            utility.Resize,
            utility.Resolution,
        ]

    @rowproperty
    def video(self):
        from foundation.utility.resize import Resize, Resolution
        from foundation.stimulus.video import Video

        # load video
        video = (Video & self.item).link.compute.video

        # target size
        height, width = (Resolution & self.item).fetch1("height", "width")

        # resize video
        return (Resize & self.item).link.resize(video, height, width)
