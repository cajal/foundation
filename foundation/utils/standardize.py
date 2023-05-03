import numpy as np


# ------- Standardize Base -------


class Standardize:
    """Standardize Base"""

    def __init__(self, homogeneous):
        """
        Parameters
        ----------
        homogeneous : 1D array -- [units]
            boolean mask, indicates whether transformation must be homogeneous
        """
        self.homogeneous = np.array(homogeneous, dtype=bool)
        assert self.homogeneous.ndim == 1

    @property
    def units(self):
        return self.homogeneous.size

    def __call__(self, a):
        """
        Parameters
        ----------
        a : 2D array
            values to be standardized -- [samples, units]

        Returns
        -------
        2D array
            standardized values -- [samples, units]
        """
        raise NotImplementedError()


# ------- Standardize Types -------


class Affine(Standardize):
    def __init__(self, shift, scale, homogeneous, eps=1e-4):
        """
        Parameters
        ----------
        shift : 1D array -- [units]
            affine shift
        scale : 1D array -- [units]
            affine scale
        homogeneous : 1D array -- [units]
            boolean mask, indicates whether transformation must be homogeneous
        """
        super().__init__(homogeneous)

        self.shift = np.array(shift, dtype=float) * np.logical_not(self.homogeneous)
        self.scale = np.array(scale, dtype=float)

        assert self.shift.ndim == self.scale.ndim == 1
        assert self.shift.size == self.scale.size == self.units
        assert self.scale.min() > eps

    def __call__(self, a):
        return (a - self.shift) / self.scale


class Scale(Standardize):
    def __init__(self, scale, homogeneous, eps=1e-4):
        """
        Parameters
        ----------
        scale : 1D array -- [units]
            affine scale
        homogeneous : 1D array -- [units]
            boolean mask, indicates whether transformation must be homogeneous
        """
        super().__init__(homogeneous)

        self.scale = np.array(scale, dtype=float)

        assert self.scale.ndim == 1
        assert self.scale.size == self.units
        assert np.min(self.scale) > eps

    def __call__(self, a):
        return a / self.scale
