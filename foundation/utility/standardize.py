import numpy as np
from djutils import row_property, row_method
from foundation.utility import stat
from foundation.schemas import utility as schema


# ---------- Standardize ----------

# -- Standardize Base --


class _Standardize:
    """Standardize using Summary statistics"""

    @row_property
    def summary_keys(self):
        """
        Returns
        -------
        foundation.utility.stat.SummaryLink
            tuples
        """
        raise NotImplementedError()

    @row_method
    def standardizer(self, homogeneous, **kwargs):
        """
        Parameter
        ---------
        homogeneous : 1D array
            boolean mask -- whether transformation must be homogeneous -- [size]
        **kwargs
            summary statistic key : 1D array
                summary statistic value -- [size]

        Returns
        -------
        foundation.utils.standardize.Standardize
            standardized array -- [samples, size]
        """
        raise NotImplementedError()


# -- Standardize Base --


@schema.lookup
class Affine(_Standardize):
    definition = """
    -> stat.SummaryLink.proj(summary_id_shift="summary_id")
    -> stat.SummaryLink.proj(summary_id_scale="summary_id")
    """

    @row_property
    def summary_keys(self):
        return stat.SummaryLink & [dict(summary_id=v) for v in self.fetch1().values()]

    @row_method
    def standardizer(self, homogeneous, **kwargs):
        from foundation.utils.standardize import Affine

        shift_key, scale_key = self.fetch1("summary_id_shift", "summary_id_scale")
        shift = kwargs[shift_key]
        scale = kwargs[scale_key]

        return Affine(shift=shift, scale=scale, homogeneous=homogeneous)


@schema.lookup
class Scale(_Standardize):
    definition = """
    -> stat.SummaryLink
    """

    @row_property
    def summary_keys(self):
        return stat.SummaryLink & self

    @row_method
    def standardizer(self, homogeneous, **kwargs):
        from foundation.utils.standardize import Scale

        key = self.fetch1("summary_id")
        scale = kwargs[key]

        return Scale(scale=scale, homogeneous=homogeneous)


# -- Standardize Link --


@schema.link
class StandardizeLink:
    links = [Affine, Scale]
    name = "standardize"
    comment = "standardization method"
