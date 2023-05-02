from djutils import row_property, row_method
from foundation.utils.video import Image, Video
from foundation.utils.logging import logger
from foundation.schemas import utility as schema


# ---------- Resolution ----------


@schema.lookup
class Resolution:
    definition = """
    height      : int unsigned  # height (pixels)
    width       : int unsigned  # width (pixels)
    """


# ---------- Resize ----------

# -- Resize Base --


class _Resize:
    """Video Resizing"""

    @row_method
    def resize(self, video, height, width):
        """
        Parameters
        -------
        video : Video
            trace times, monotonically increasing
        height : int
            target height (pixels)
        width : int
            target width (pixels)

        Returns
        -------
        Video
            resized video
        """
        raise NotImplementedError()


# -- Resize Types --


@schema.lookup
class PilResample(_Resize):
    definition = """
    resample        : varchar(64)   # resampling filter (PIL.Image.Resampling)
    """

    @row_method
    def resize(self, video, height, width):

        if height == video.height and width == video.width:
            return video

        else:
            resample = getattr(Image.Resampling, self.fetch1("resample"))
            resize = lambda frame: frame.resize(size=(width, height), resample=resample)
            return video.apply(resize)


# -- Resize Link --


@schema.link
class ResizeLink:
    links = [PilResample]
    name = "resize"
    comment = "video resizing"
