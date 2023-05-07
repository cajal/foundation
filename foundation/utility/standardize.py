import numpy as np
from djutils import rowproperty, rowmethod
from foundation.utility.stat import SummaryLink
from foundation.schemas import utility as schema


# ---------- Standardize ----------

# -- Standardize Base --


class _Standardize:
    """Standardize using Summary statistics"""

    @rowproperty
    def summary_keys(self):
        """
        Returns
        -------
        foundation.utility.stat.SummaryLink
            tuples
        """
        raise NotImplementedError()

    @rowmethod
    def standardize(self, homogeneous, **kwargs):
        """
        Parameter
        ---------
        homogeneous : 1D array
            boolean mask -- whether transformation must be homogeneous -- [size]
        **kwargs
            key -- summary_id
            value -- 1D array -- [size]

        Returns
        -------
        foundation.utils.standardize.Standardize
            callable, standardizes data
        """
        raise NotImplementedError()


# -- Standardize Base --


@schema.lookup
class Affine(_Standardize):
    definition = """
    -> SummaryLink.proj(summary_id_shift="summary_id")
    -> SummaryLink.proj(summary_id_scale="summary_id")
    """

    @rowproperty
    def summary_keys(self):
        return SummaryLink & [dict(summary_id=v) for v in self.fetch1().values()]

    @rowmethod
    def standardize(self, homogeneous, **kwargs):
        from foundation.utils.standardize import Affine

        shift_key, scale_key = self.fetch1("summary_id_shift", "summary_id_scale")
        shift = kwargs[shift_key]
        scale = kwargs[scale_key]

        return Affine(shift=shift, scale=scale, homogeneous=homogeneous)


@schema.lookup
class Scale(_Standardize):
    definition = """
    -> SummaryLink
    """

    @rowproperty
    def summary_keys(self):
        return SummaryLink & self

    @rowmethod
    def standardize(self, homogeneous, **kwargs):
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
