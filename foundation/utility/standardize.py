import numpy as np
from djutils import row_method
from foundation.schemas import utility as schema


# ---------- Standardize ----------

# -- Standardize Base --


class _Standardize:
    """Summary Statistic"""

    @row_property
    def summary_statistics(self):
        """
        Returns
        -------
        foundation.utility.stat.SummaryLink
            tuples
        """
        raise NotImplementedError()

    @row_method
    def standardize(self, a, homogeneous, **kwargs):
        """
        Parameter
        ---------
        a : 2D array
            array of values -- [samples, size]
        homogeneous : 1D array
            boolean mask -- whether transformation must be homogeneous -- [size]
        **kwargs
            summary statistic key : 1D array
                summary statistic value -- [size]

        Returns
        -------
        2D array
            standardized array
        """
        raise NotImplementedError()
