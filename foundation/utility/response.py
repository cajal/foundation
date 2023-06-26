from djutils import rowproperty
from foundation.schemas import utility as schema


# ---------------------------- Response Burnin ----------------------------


@schema.lookup
class Burnin:
    definition = """
    burnin      : int unsigned      # response burnin -- discarded initial frames
    """


# ---------------------------- Response Measure ----------------------------

# -- Response Measure Base --


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


# -- Response Measure Types --


@schema.method
class CCMax(MeasureType):
    name = "cc_max"
    comment = "upper bound of signal correlation"

    @rowproperty
    def measure(self):
        from foundation.utils.response import CCMax

        return CCMax()


# -- Response Measure --


@schema.link
class Measure:
    links = [CCMax]
    name = "measure"
    comment = "response measure"


# ---------------------------- Response Correlation ----------------------------

# -- Response Correlation Base --


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


# -- Response Measure Types --


@schema.method
class CCSignal(CorrelationType):
    name = "cc_signal"
    comment = "signal correlation"

    @rowproperty
    def correlation(self):
        from foundation.utils.response import CCSignal

        return CCSignal()


# -- Response Measure --


@schema.link
class Correlation:
    links = [CCSignal]
    name = "correlation"
    comment = "response correlation"
