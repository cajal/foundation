from djutils import keys, rowproperty
from foundation.utility.resize import ResizeLink, Resolution
from foundation.stimulus.video import Video


@keys
class ResizeVideo:
    """Resize video"""

    @property
    def key_list(self):
        return [
            Video,
            ResizeLink,
            Resolution,
        ]

    @rowproperty
    def video(self):
        # load video
        video = (Video & self.key).link.video

        # target size
        height, width = (Resolution & self.key).fetch1("height", "width")

        # resize video
        return (ResizeLink & self.key).link.resize(video, height, width)
