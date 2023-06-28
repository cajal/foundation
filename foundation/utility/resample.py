from djutils import rowproperty, rowmethod
from foundation.schemas import utility as schema


# ---------------------------- Rate ----------------------------

# -- Rate Interface --


class RateType:
    """Resampling Rate"""

    @rowproperty
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
class Hz(RateType):
    definition = """
    hz          : decimal(9, 6)         # samples per second
    """

    @rowproperty
    def period(self):
        return 1 / float(self.fetch1("hz"))


# -- Rate --


@schema.link
class Rate:
    links = [Hz]
    name = "rate"
    comment = "resampling rate"


# ---------------------------- Offset ----------------------------

# -- Offset Interface --


class OffsetType:
    """Resampling Offset"""

    @rowproperty
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
class MsOffset(OffsetType):
    definition = """
    ms_offset       : int unsigned      # millisecond offset
    """

    @rowproperty
    def offset(self):
        return self.fetch1("ms_offset") / 1000


# -- Offset --


@schema.link
class Offset:
    links = [MsOffset]
    name = "offset"
    comment = "resampling offset"


# ---------------------------- Resample ----------------------------

# -- Resample Interface --


class ResampleType:
    """Resampling Method"""

    @rowmethod
    def resample(self, times, values, target_period, target_offset):
        """
        Parameters
        -------
        times : 1D array
            trace times, monotonically increasing
        values : 1D array
            trace values, same length as times
        target_period : float
            target sampling period
        target_offset : float
            target sampling offset

        Returns
        -------
        foundation.utils.resample.Resample
            callable, resamples traces
        """
        raise NotImplementedError()


# -- Resample Types --


@schema.method
class Hamming(ResampleType):
    name = "hamming"
    comment = "hamming trace"

    @rowmethod
    def resample(self, times, values, target_period, target_offset):
        from foundation.utils.resample import Hamming

        return Hamming(
            times=times,
            values=values,
            target_period=target_period,
            target_offset=target_offset,
        )


@schema.lookup
class LowpassHamming(ResampleType):
    definition = """
    lowpass_hz      : decimal(6, 3)     # lowpass filter rate
    """

    @rowmethod
    def resample(self, times, values, target_period, target_offset):
        from foundation.utils.resample import LowpassHamming

        return LowpassHamming(
            times=times,
            values=values,
            target_period=target_period,
            lowpass_period=1 / float(self.fetch1("lowpass_hz")),
            target_offset=target_offset,
        )


# -- Resample --


@schema.link
class Resample:
    links = [Hamming, LowpassHamming]
    name = "resample"
    comment = "resampling method"
