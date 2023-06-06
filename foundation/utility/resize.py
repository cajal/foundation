from djutils import rowproperty
from foundation.utils import resize
from foundation.schemas import utility as schema


# ---------------------------- Resolution ----------------------------


@schema.lookup
class Resolution:
    definition = """
    height      : int unsigned  # height (pixels)
    width       : int unsigned  # width (pixels)
    """


# ---------------------------- Resize ----------------------------

# -- Resize Base --


class _Resize:
    """Video Resizing"""

    @rowproperty
    def resize(self):
        """
        Returns
        -------
        foundation.utils.resize.Resize
            callable, resizes videos
        """
        raise NotImplementedError()


# -- Resize Types --


@schema.lookup
class PilResize(_Resize):
    definition = """
    resample        : varchar(64)   # resampling filter (PIL.Image.Resampling)
    """

    @rowproperty
    def resize(self):
        from PIL import Image

        resample = getattr(Image.Resampling, self.fetch1("resample"))
        return resize.PilResize(resample)


# -- Resize --


@schema.link
class Resize:
    links = [PilResize]
    name = "resize"
    comment = "resizing method"
