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


def spline(x, y, k=1, ext=2, nan=False, ntol=0):
    """1-D interpolative spline

    Parameters
    ----------
    x : np.array
        [N] -- must be strictly increasing
    y : np.array
        [N +/- ntol]
    k : int
        degree of the smoothing spline -- must be 1 <= k <= 5
    ext : int | str
        controls the extrapolation mode for elements not in the interval defined by the knot sequence.
            if ext=0 or 'extrapolate', return the extrapolated value
            if ext=1 or 'zeros', return 0
            if ext=2 or 'raise', raise a ValueError
            if ext=3 of 'const', return the boundary value
    ntol : int
        tolerance for x and y length mismatch

    Returns
    -------
    InterpolatedUnivariateSpline
    """
    nx = len(x)
    ny = len(y)

    if abs(nx - ny) > ntol:
        raise ValueError(f"x and y differ in length by more than {ntol}")

    if nx > ny:
        logger.warning(f"Truncating x by {nx - ny}")
        x = x[:ny]

    elif ny > nx:
        logger.warning(f"Truncating y by {ny - nx}")
        y = y[:nx]

    if nan:
        return NaNSpline(x, y, k=k, ext=ext)
    else:
        assert np.isfinite(x).all()
        assert np.isfinite(y).all()
        return CenteredSpline(x, y, k=k, ext=ext)
