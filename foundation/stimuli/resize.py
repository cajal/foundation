import numpy as np
import datajoint as dj
from djutils import link
from PIL import Image
from foundation.utils.logging import logger


schema = dj.schema("foundation_stimuli")


class ResizeMixin:
    @property
    def _resize(self):
        """
        Returns
        -------
        Callable[[image: Image, height: int, width: int], Image]
        """
        raise NotImplementedError()

    @property
    def resize(self):
        f = self._resize

        def resize(image, height, width):
            """
            Parameters
            ----------
            image : Image
                stimulus frame
            height : int
                target height
            width : int
                target width

            Returns
            -------
            Image
                resized stimulus frame
            """
            if height == image.height and width == image.width:
                return image
            else:
                return f(image, height, width)

        return resize


@schema
class ResampleFilter(dj.Lookup, ResizeMixin):
    definition = """
    resample_filter     : varchar(32)   # resampling filter
    """

    @property
    def _resize(self):
        filt = self.fetch1("resample_filter")
        resample = getattr(Image, filt)
        logger.info(f"Resampling with {filt} filter")

        def f(image, height, width):
            return image.resize((width, height), resample=resample)

        return f


@link(schema)
class Resize:
    links = [ResampleFilter]
    name = "resize"
    comment = "stimulus resizing method"
    length = 8
