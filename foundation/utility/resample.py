from djutils import row_property, row_method
from foundation.schemas import utility as schema


# ---------- Rate ----------

# -- Rate Base --


class _Rate:
    """Resampling Rate"""

    @row_property
    def period(self):
        """
        Returns
        -------
        float
            sampling period (seconds)
        """
        raise NotImplementedError()


# -- Rate Types --


@schema.lookup
class Hz(_Rate):
    definition = """
    hz          : decimal(9, 6)         # samples per second
    """

    @row_property
    def period(self):
        return 1 / float(self.fetch1("hz"))


# -- Rate Link --


@schema.link
class RateLink:
    links = [Hz]
    name = "rate"
    comment = "resampling rate"


# ---------- Offset ----------

# -- Offset Base --


class _Offset:
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


@schema.method
class ZeroOffset(_Offset):
    name = "zero_offset"
    comment = "zero resampling offset"

    @row_property
    def offset(self):
        return 0


# -- Offset Link --


@schema.link
class OffsetLink:
    links = [ZeroOffset]
    name = "offset"
    comment = "resampling offset"


# ---------- Resample ----------

# -- Resample Base --


class _Resample:
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


@schema.method
class Hamming(_Resample):
    name = "hamming"
    comment = "hamming trace"

    @row_method
    def resampler(self, times, values, target_period):
        from foundation.utils.trace import Hamming

        return Hamming(times, values, target_period)


# -- Resample Link --


@schema.link
class ResampleLink:
    links = [Hamming]
    name = "resample"
    comment = "trace resampling"
