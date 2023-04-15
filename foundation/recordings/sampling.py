import datajoint as dj
from djutils import link


schema = dj.schema("foundation_recordings", context=locals())


# ---------- Sampling Rates -------------


class RateMixin:
    @property
    def period(self):
        """
        Returns
        -------
        float
            sampling period (seconds)
        """
        raise NotImplementedError()


@schema
class Hz(dj.Lookup, RateMixin):
    definition = """
    hz          : decimal(9, 6)         # samples per second
    """

    @property
    def period(self):
        return 1 / float(self.fetch1("hz"))


@link(schema)
class Rate:
    links = [Hz]
    name = "rate"
    comment = "sampling rate"
    length = 8


# ---------- Sampling Offsets -------------


class OffsetMixin:
    @property
    def offset(self):
        """
        Returns
        -------
        float
            sampling offset (seconds)
        """
        raise NotImplementedError()


@schema
class Frames(dj.Lookup, OffsetMixin):
    definition = """
    -> Rate
    frames      : smallint unsigned     # number of offset frames
    """

    @property
    def offset(self):
        period = (Rate & self).link.period
        frames = self.fetch1("frames")
        return period * frames


@link(schema)
class Offset:
    links = [Frames]
    name = "offset"
    comment = "sampling offset"
    length = 8
