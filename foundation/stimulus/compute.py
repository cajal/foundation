from djutils import keys, rowproperty
from foundation.virtual import utility, stimulus


@keys
class ResizeVideo:
    """Resize video"""

    @property
    def key_list(self):
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
        video = (Video & self.key).link.video

        # target size
        height, width = (Resolution & self.key).fetch1("height", "width")

        # resize video
        return (Resize & self.key).link.resize(video, height, width)
