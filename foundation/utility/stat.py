import numpy as np
import datajoint as dj
from djutils import link, method, row_method
from foundation.schemas import utility as schema


# ---------- Summary ----------

# -- Summary Base --


class SummaryBase:
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


@method(schema)
class Mean(SummaryBase):
    name = "mean"
    comment = "arithmetic mean"

    @row_method
    def stat(self, a):
        return np.mean(a)


@schema
class Std(SummaryBase, dj.Lookup):
    definition = """
    ddof        : int unsigned      # delta degrees of freedom
    """

    @row_method
    def stat(self, a):
        return np.std(a, ddof=self.fetch1("ddof"))


# -- Summary Link --


@link(schema)
class SummaryLink:
    links = [Mean, Std]
    name = "summary"
    comment = "summary statistic"
