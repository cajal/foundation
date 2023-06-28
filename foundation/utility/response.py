from djutils import rowproperty
from foundation.schemas import utility as schema


# ---------------------------- Burnin ----------------------------


@schema.lookup
class Burnin:
    definition = """
    burnin      : int unsigned      # response burnin -- discarded initial frames
    """


# ---------------------------- Measure ----------------------------

# -- Measure Interface --


class MeasureType:
    """Response Measure"""

    @rowproperty
    def measure(self):
        """
        Returns
        -------
        foundation.utils.response.Measure
            callable, computes response measure
        """
        raise NotImplementedError()


# -- Measure Types --


@schema.method
class CCMax(MeasureType):
    name = "cc_max"
    comment = "upper bound of signal correlation"

    @rowproperty
    def measure(self):
        from foundation.utils.response import CCMax

        return CCMax()


# -- Measure --


@schema.link
class Measure:
    links = [CCMax]
    name = "measure"
    comment = "response measure"


# ---------------------------- Correlation ----------------------------

# -- Correlation Interface --


class CorrelationType:
    """Response Correlation"""

    @rowproperty
    def correlation(self):
        """
        Returns
        -------
        foundation.utils.response.Correlation
            callable, computes response correlation
        """
        raise NotImplementedError()


# -- Correlation Types --


@schema.method
class CCSignal(CorrelationType):
    name = "cc_signal"
    comment = "signal correlation"

    @rowproperty
    def correlation(self):
        from foundation.utils.response import CCSignal

        return CCSignal()


# -- Correlation --


@schema.link
class Correlation:
    links = [CCSignal]
    name = "correlation"
    comment = "response correlation"
