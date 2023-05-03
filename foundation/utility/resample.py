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


@schema.lookup
class MsOffset(_Offset):
    definition = """
    ms_offset       : int unsigned      # millisecond offset
    """

    @row_property
    def offset(self):
        return self.fetch1("ms_offset") / 1000


# -- Offset Link --


@schema.link
class OffsetLink:
    links = [MsOffset]
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
        foundation.utils.resample.Resample
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

        from foundation.utils.resample import Hamming

        return Hamming(
            times=times,
            values=values,
            target_period=target_period,
        )


@schema.lookup
class LowpassHamming(_Resample):
    definition = """
    lowpass_hz      : decimal(6, 3)     # lowpass filter rate
    """

    @row_method
    def resampler(self, times, values, target_period):

        from foundation.utils.resample import LowpassHamming

        return LowpassHamming(
            times=times,
            values=values,
            target_period=target_period,
            lowpass_period=1 / float(self.fetch1("lowpass_hz")),
        )


# -- Resample Link --


@schema.link
class ResampleLink:
    links = [Hamming, LowpassHamming]
    name = "resample"
    comment = "trace resampling"
