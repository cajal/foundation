from djutils import row_property
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

    @row_property
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
class PilResample(_Resize):
    definition = """
    resample        : varchar(64)   # resampling filter (PIL.Image.Resampling)
    """

    @row_property
    def resize(self):
        from PIL import Image
        from foundation.utils.resize import PilResize

        resample = getattr(Image.Resampling, self.fetch1("resample"))

        return PilResize(resample)


# -- Resize Link --


@schema.link
class ResizeLink:
    links = [PilResample]
    name = "resize"
    comment = "resizing method"
