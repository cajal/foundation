import numpy as np
from djutils import rowproperty
from foundation.schemas import utility as schema


# ------------------------------------ Summary ------------------------------------

# -- Summary Interface --


class SummaryType:
    """Summary Statistic"""

    @rowproperty
    def summary(self):
        """
        Returns
        -------
        Callable[[1D array], float]
            function that takes a 1D array and returns a summary statistic float
        """
        raise NotImplementedError()


# -- Summary Types --


@schema.method
class Minimum(SummaryType):
    name = "minimum"
    comment = "minimum"

    @rowproperty
    def summary(self):
        return np.min


@schema.method
class Mean(SummaryType):
    name = "mean"
    comment = "arithmetic mean"

    @rowproperty
    def summary(self):
        return np.mean


@schema.lookup
class Std(SummaryType):
    definition = """
    ddof        : int unsigned      # delta degrees of freedom
    """

    @rowproperty
    def summary(self):
        ddof = self.fetch1("ddof")
        return lambda x: np.std(x, ddof=ddof)


# -- Summary --


@schema.link
class Summary:
    links = [Minimum, Mean, Std]
    name = "summary"
    comment = "summary statistic"
