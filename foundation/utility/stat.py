import numpy as np
from djutils import row_method
from foundation.schemas import utility as schema


# ---------- Summary ----------

# -- Summary Base --


class Summary:
    """Summary Statistic"""

    @row_method
    def stat(self, a):
        """
        Parameter
        ---------
        a : 1D array
            array of values

        Returns
        -------
        float
            summary statistic
        """
        raise NotImplementedError()


# -- Summary Type --


@schema.method
class Mean(Summary):
    name = "mean"
    comment = "arithmetic mean"

    @row_method
    def stat(self, a):
        return np.mean(a)


@schema.lookup
class Std(Summary):
    definition = """
    ddof        : int unsigned      # delta degrees of freedom
    """

    @row_method
    def stat(self, a):
        return np.std(a, ddof=self.fetch1("ddof"))


# -- Summary Link --


@schema.link
class SummaryLink:
    links = [Mean, Std]
    name = "summary"
    comment = "summary statistic"
