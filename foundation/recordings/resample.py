import datajoint as dj
from djutils import link, method, row_property


schema = dj.schema("foundation_recordings")


# ---------- Rate ----------

# -- Rate Base --


class RateBase:
    """Resampling Rate"""

    @row_property
    def period(self):
        """
        Returns
        -------
        float
            resampling period (seconds)
        """
        raise NotImplementedError()


# -- Rate Types --


@schema
class Hz(RateBase, dj.Lookup):
    definition = """
    hz          : decimal(9, 6)         # samples per second
    """

    @row_property
    def period(self):
        return 1 / float(self.fetch1("hz"))


# -- Rate Link --


@link(schema)
class RateLink:
    links = [Hz]
    name = "rate"
    comment = "resampling rate"


# -- Computed Rate --


@schema
class Rate(RateBase, dj.Computed):
    definition = """
    -> RateLink
    ---
    resample_period     : double        # resampling period (seconds)
    """

    def make(self, key):
        key["resample_period"] = (RateLink & key).link.period
        self.insert1(key)

    @row_property
    def period(self):
        return self.fetch1("resample_period")


# ---------- Offset ----------

# -- Offset Base --


class OffsetBase:
    """Resampling Offset"""

    @row_property
    def offset(self):
        """
        Returns
        -------
        float
            sampling offset (seconds)
        """
        raise NotImplementedError()


# -- Offset Types --


@method(schema)
class ZeroOffset(OffsetBase):
    name = "zero_offset"
    comment = "zero resampling offset"

    @row_property
    def offset(self):
        return 0


# -- Offset Link --


@link(schema)
class OffsetLink:
    links = [ZeroOffset]
    name = "offset"
    comment = "resampling offset"


# -- Computed Offset --


@schema
class Offset(OffsetBase, dj.Computed):
    definition = """
    -> OffsetLink
    ---
    resample_offset     : double        # resampling offset (seconds)
    """

    def make(self, key):
        key["resample_offset"] = (OffsetLink & key).link.offset
        self.insert1(key)

    @row_property
    def offset(self):
        return self.fetch1("resample_offset")
