import datajoint as dj
from djutils import link


schema = dj.schema("foundation_recordings")


# ---------- Sampling Rates ----------


class RateBase:
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
class Hz(RateBase, dj.Lookup):
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


# ---------- Sampling Offsets ----------


class OffsetBase:
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
class OffsetFrames(OffsetBase, dj.Lookup):
    definition = """
    -> Rate
    offset_frames   : smallint unsigned     # number of offset frames
    """

    @property
    def offset(self):
        period = (Rate & self).link.period
        frames = self.fetch1("offset_frames")
        return period * frames


@link(schema)
class Offset:
    links = [OffsetFrames]
    name = "offset"
    comment = "sampling offset"
