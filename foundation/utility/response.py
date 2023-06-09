from djutils import rowproperty
from foundation.utils import response
from foundation.schemas import utility as schema


# ---------------------------- Response Burnin ----------------------------


@schema.lookup
class Burnin:
    definition = """
    burnin      : int unsigned      # response burnin -- discarded initial frames
    """


# ---------------------------- Response Measure ----------------------------

# -- Response Measure Base --


class _Measure:
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
class CCMax(_Measure):
    name = "cc_max"
    comment = "upper bound of signal correlation"

    @rowproperty
    def measure(self):
        return response.CCMax()


# -- Response Measure --


@schema.link
class Measure:
    links = [CCMax]
    name = "measure"
    comment = "response measure"


# ---------------------------- Response Correlation ----------------------------

# -- Response Correlation Base --


class _Correlation:
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
class CCSignal(_Correlation):
    name = "cc_signal"
    comment = "signal correlation"

    @rowproperty
    def correlation(self):
        return response.CCSignal()


# -- Response Measure --


@schema.link
class Correlation:
    links = [CCSignal]
    name = "correlation"
    comment = "response correlation"
