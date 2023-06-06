from djutils import rowproperty
from foundation.utils import measure
from foundation.schemas import utility as schema


# ---------------------------- Measure ----------------------------

# -- Measure Base --


class _Measure:
    """Response Measure"""

    @rowproperty
    def measure(self):
        """
        Returns
        -------
        foundation.utils.measure.Measure
            callable, computes response measure
        """
        raise NotImplementedError()


# -- Measure Types --


@schema.method
class CCMax(_Measure):
    name = "cc_max"
    comment = "upper bound of signal correlation"

    @rowproperty
    def measure(self):
        return measure.CCMax()


# -- Measure --


@schema.link
class Measure:
    links = [CCMax]
    name = "measure"
    comment = "functional measure"
