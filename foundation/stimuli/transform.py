import numpy as np
import datajoint as dj
from djutils import link
from foundation.utils.video import Video
from foundation.utils.logging import logger


schema = dj.schema("foundation_stimuli")


# ---------- Transform Mixin ----------


class TransformMixin:
    @property
    def _tranform(self):
        """
        Returns
        -------
        Callable[[Video], Video]
        """
        raise NotImplementedError()

    @property
    def transform(self):
        f = self._tranform

        def transform(frames):
            """
            Parameters
            ----------
            frames : Video
                stimulus frames

            Returns
            -------
            Video
                resized stimulus frames
            """
            return f(frames)

        return transform


# ---------- Resize ----------


@schema
class Resize(TransformMixin, dj.Lookup):
    definition = """
    height      : smallint unsigned     # stimulus frame height
    width       : smallint unsigned     # stimulus frame width
    resample    : varchar(64)           # resampling filter
    """

    @property
    def _tranform(self):
        from PIL.Image import BILINEAR

        filters = {
            "bilinear": BILINEAR,
        }

        height, width, resample = self.fetch1("height", "width", "resample")

        if resample not in filters:
            raise NotImplementedError(f"Resample type '{resample}' has not been implemented")
        else:
            filt = filters[resample]

        def f(frames):

            if height == frames.height and width == frames.width:
                logger.info(f"Frame size is already {height}x{width}. Not resampling.")
                return frames

            else:
                logger.info(f"Resampling to {height}x{width} with {resample} filter.")
                resize = lambda frame: frame.resize((width, height), resample=filt)
                return frames.apply(resize)

        return f


# ---------- Transform Links ----------


@link(schema)
class Transform:
    links = [Resize]
    name = "transform"
    comment = "stimulus transformation"
    length = 16
