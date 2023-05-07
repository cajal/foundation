from djutils import keys, rowproperty
from foundation.utility.resize import ResizeLink, Resolution
from foundation.stimulus.video import VideoLink


@keys
class ResizeVideo:
    """Resize video"""

    @property
    def key_list(self):
        return [
            VideoLink,
            ResizeLink,
            Resolution,
        ]

    @rowproperty
    def video(self):
        # load video
        video = (VideoLink & self.key).link.video

        # target size
        height, width = (Resolution & self.key).fetch1("height", "width")

        # resize video
        return (ResizeLink & self.key).link.resize(video, height, width)
