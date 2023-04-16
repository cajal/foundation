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
        Callable[[Image], Image]
        """
        raise NotImplementedError()

    @property
    def resize(self):
        f = self._resize

        def resize(image):
            """
            Parameters
            ----------
            image : Image
                stimulus frame

            Returns
            -------
            Image
                resized stimulus frame
            """
            return f(image)

        return resize


@schema
class Bilinear(dj.Lookup, ResizeMixin):
    definition = """
    height      : smallint unsigned  # stimulus frame height
    width       : smallint unsigned  # stimulus frame width
    """

    @property
    def _resize(self):
        height, width = self.fetch1("height", "width")

        def f(image):
            if image.height == height and image.width == width:
                return image
            else:
                return image.resize((width, height), resample=Image.BILINEAR)

        return f


@link(schema)
class Resize:
    links = [Bilinear]
    name = "resize"
    comment = "stimulus resizing method"
    length = 8
