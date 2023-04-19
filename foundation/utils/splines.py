import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from .logging import logger


class CenteredSpline(InterpolatedUnivariateSpline):
    def __init__(self, x, y, **kwargs):
        # center x for numerical stability
        self.x0 = np.nanmedian(x)
        x = x - self.x0
        super().__init__(x, y, **kwargs)

    def __call__(self, x, **kwargs):
        # transform x to centerered position
        x = x - self.x0
        return super().__call__(x, **kwargs)


class NaNSpline(InterpolatedUnivariateSpline):
    def __init__(self, x, y, **kwargs):
        # center x for numerical stability
        self.x0 = np.nanmedian(x)
        x = x - self.x0

        # detect nans in x
        xnan = np.isnan(x)
        if np.any(xnan):
            logger.warning(f"Found {xnan.sum()} nans in the x-values")
            x[xnan] = np.interp(np.where(xnan)[0], np.where(~xnan)[0], x[~xnan])

        # detect nans in y
        ynan = np.isnan(y)
        if np.any(ynan):
            logger.warning(f"Found {ynan.sum()} nans in the y-values")
            y = y.copy()
            y[ynan] = 0

        # finite spline
        nan = xnan | ynan
        super().__init__(x[~nan], y[~nan], **kwargs)

        # nan spline (0 inidcates finite, > 0 indicates nan)
        self.nans = interp1d(x, 1 * nan, kind="linear")

    def __call__(self, x, **kwargs):
        # transform x to centerered position
        x = x - self.x0

        # nan mask
        notnan = ~np.isnan(x)
        nan = np.ones_like(x)
        nan[notnan] = self.nans(x[notnan])
        nan = nan > 0

        # interpolate non-nan values, nan otherwise
        ret = np.full_like(x, np.nan)
        ret[~nan] = super().__call__(x[~nan], **kwargs)

        return ret
