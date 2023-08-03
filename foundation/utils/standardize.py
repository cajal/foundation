import numpy as np


# ------- Standardize Interface -------


class Standardize:
    """Standardize Base"""

    def __init__(self, homogeneous):
        """
        Parameters
        ----------
        homogeneous : 1D array
            [N] -- dtype=bool -- homogeneous | unrestricted transform
        """
        self.homogeneous = np.array(homogeneous, dtype=bool)
        assert self.homogeneous.ndim == 1

    def __len__(self):
        return self.homogeneous.size

    def __call__(self, a, inverse=False):
        """
        Parameters
        ----------
        a : 2D array
            [M, N] -- dtype=float -- values to be transformed
        inverse : bool
            inverse | normal transform

        Returns
        -------
        2D array
            [M, N] -- dtype=float -- transformed values
        """
        raise NotImplementedError()


# ------- Standardize Types -------


class Affine(Standardize):
    """Affine Transform"""

    def __init__(self, shift, scale, homogeneous, eps=1e-4):
        """
        Parameters
        ----------
        shift : 1D array
            [N] -- dtype=float -- affine shift
        scale : 1D array
            [N] -- dtype=float -- affine scale
        homogeneous : 1D array
            [N] -- dtype=bool -- homogeneous | unrestricted transform
        eps : float
            small number for numerical stability
        """
        super().__init__(homogeneous)

        self.eps = float(eps)
        self.shift = np.array(shift, dtype=float) * np.logical_not(self.homogeneous)
        self.scale = np.array(scale, dtype=float).clip(min=self.eps)

        assert self.shift.ndim == self.scale.ndim == 1
        assert self.shift.size == self.scale.size == len(self)

    def __call__(self, a, inverse=False):
        if inverse:
            return a * self.scale + self.shift
        else:
            return (a - self.shift) / self.scale


class Scale(Standardize):
    """Scale Transform"""

    def __init__(self, scale, homogeneous, eps=1e-4):
        """
        Parameters
        ----------
        scale : 1D array
            [N] -- dtype=float -- divisive scale
        homogeneous : 1D array
            [N] -- dtype=bool -- homogeneous | unrestricted transform
        eps : float
            small number for numerical stability
        """
        super().__init__(homogeneous)

        self.eps = float(eps)
        self.scale = np.array(scale, dtype=float).clip(min=self.eps)

        assert self.scale.ndim == 1
        assert self.scale.size == len(self)

    def __call__(self, a, inverse=False):
        if inverse:
            return a * scale
        else:
            return a / self.scale
