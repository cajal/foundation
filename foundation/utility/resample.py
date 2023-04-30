import datajoint as dj
from djutils import link, method, row_property, row_method
from foundation.schemas import utility as schema


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


# ---------- Resample ----------

# -- Resample Base --


class ResampleBase:
    """Trace Resampling"""

    @row_method
    def resampler(self, times, values, target_period):
        """
        Parameters
        -------
        times : 1D array
            trace times, monotonically increasing
        values : 1D array
            trace values, same length as times
        target_period : float
            target sampling period

        Returns
        -------
        foundation.utils.trace.Trace
            trace resampler
        """
        raise NotImplementedError()


# -- Resample Types --


@method(schema)
class Hamming(ResampleBase):
    name = "hamming"
    comment = "hamming trace"

    @row_method
    def resampler(self, times, values, target_period):
        from foundation.utils.trace import Hamming

        return Hamming(times, values, target_period)


# -- Resample Link --


@link(schema)
class ResampleLink:
    links = [Hamming]
    name = "resample"
    comment = "trace resampling"
