import numpy as np
from djutils import rowproperty, keys

from foundation.virtual import recording, fnn, tuning


# ----------------------------- Spatial Tuning -----------------------------

# -- Spatial Tuning Interface --


class SpatialType:
    """Spatial Response"""

    @rowproperty
    def tuning(self):
        """
        Returns
        -------
        2D array
            response to spatial locations
        """
        raise NotImplementedError()


# -- Spatial Tuning Types --


@keys
class RecordingVisualSpatial(SpatialType):
    """Recording Visual Spatial Tuning"""

    @property
    def keys(self):
        return [
            recording.VisualSpatialTuning,
        ]

    @rowproperty
    def tuning(self):
        return (recording.VisualSpatialTuning & self.item).fetch1("response")


@keys
class FnnVisualSpatial(SpatialType):
    """Fnn Visual Spatial Tuning"""

    @property
    def keys(self):
        return [
            fnn.VisualSpatialTuning,
        ]

    @rowproperty
    def tuning(self):
        return (fnn.VisualSpatialTuning & self.item).fetch1("response")


# ----------------------------- Spatial Fit -----------------------------


def grid(height, width):
    size = max(height, width)
    x = np.arange(width) / size
    y = np.arange(height) / size
    return x, y


def meshgrid(height, width):
    x, y = grid(height, width)
    x = np.linspace(0, x.max(), width)
    y = np.linspace(0, y.max(), height)
    return np.meshgrid(x, y, indexing="xy")


@keys
class SSI:
    """Spatial Selectivity Index"""

    @property
    def keys(self):
        return [
            tuning.Spatial,
        ]

    @rowproperty
    def ssi(self):
        """
        Returns
        -------
        float
            spatial selectivity index
        """
        from foundation.tuning.spatial import Spatial

        rf = (Spatial & self.item).link.compute.tuning

        x, y = meshgrid(*rf.shape)
        z = rf / rf.sum()

        mu_x = (z * x).sum()
        mu_y = (z * y).sum()

        _x = x - mu_x
        _y = y - mu_y

        cov_xx = (z * _x * _x).sum()
        cov_xy = (z * _x * _y).sum()
        cov_yy = (z * _y * _y).sum()

        return -np.log(cov_xx * cov_yy - cov_xy**2)
