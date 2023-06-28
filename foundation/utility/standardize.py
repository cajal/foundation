import numpy as np
from djutils import rowproperty, rowmethod
from foundation.utility.stat import Summary
from foundation.schemas import utility as schema


# ---------------------------- Standardize ----------------------------

# -- Standardize Interface --


class StandardizeType:
    """Standardization Method"""

    @rowproperty
    def summary_ids(self):
        """
        Returns
        -------
        List[str]
            list of keys (foundation.utility.stat.Summary)
        """
        raise NotImplementedError()

    @rowmethod
    def standardize(self, homogeneous, **kwargs):
        """
        Parameter
        ---------
        homogeneous : 1D array
            [size] -- dtype=bool -- homogeneous | any transform
        **kwargs
            summary_id -- key (foundation.utility.stat.Summary)
                1D array -- [size] -- dtype=float -- summary statistic

        Returns
        -------
        foundation.utils.standardize.Standardize
            callable, standardizes data
        """
        raise NotImplementedError()


# -- Standardize Types --


@schema.lookup
class Affine(StandardizeType):
    definition = """
    -> Summary.proj(summary_id_shift="summary_id")
    -> Summary.proj(summary_id_scale="summary_id")
    """

    @rowproperty
    def summary_ids(self):
        return list(self.fetch1().values())

    @rowmethod
    def standardize(self, homogeneous, **kwargs):
        from foundation.utils.standardize import Affine

        shift_key, scale_key = self.fetch1("summary_id_shift", "summary_id_scale")
        shift = kwargs[shift_key]
        scale = kwargs[scale_key]

        return Affine(shift=shift, scale=scale, homogeneous=homogeneous)


@schema.lookup
class Scale(StandardizeType):
    definition = """
    -> Summary
    """

    @rowproperty
    def summary_ids(self):
        return [self.fetch1("summary_id")]

    @rowmethod
    def standardize(self, homogeneous, **kwargs):
        from foundation.utils.standardize import Scale

        key = self.fetch1("summary_id")
        scale = kwargs[key]

        return Scale(scale=scale, homogeneous=homogeneous)


# -- Standardize --


@schema.link
class Standardize:
    links = [Affine, Scale]
    name = "standardize"
    comment = "standardization method"
