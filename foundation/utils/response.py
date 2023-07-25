import numpy as np
import pandas as pd
from scipy import stats
from .resample import truncate


# ---------------------------- Response Trials ----------------------------


class Trials(pd.Series):
    """Response Trials"""

    def __init__(self, data, index=None, tolerance=0):
        """
        Parameters
        ----------
        data : Sequence[1D array]
            trial responses
        index : Sequence[str] | None
            trial identifiers. optional if len(data) == 1
        tolerance : int
            response length mismatch tolerance
        """
        # truncate trial responses to the same length
        data = truncate(*data, tolerance=tolerance)

        # assert all responses are 1D
        assert all(map(lambda x: x.ndim == 1, data)), "Each response must be 1D."

        # initialize
        super().__init__(data=data, index=[None] if index is None else index)

    def to_array(self, size=None):
        """
        Parameters
        ----------
        size : int | None
            number of trials

        Returns
        -------
        2D array
            [trials, samples], nan-filled to desired trial size
        """
        # 2D response array
        array = np.stack(self.values, axis=0)
        trials, samples = array.shape

        if size is None or size == trials:
            # response array
            return array

        elif size > array.shape[0]:
            # nan-filled response array
            nans = np.full([size - trials, samples], np.nan, dtype=array.dtype)
            return np.concatenate([array, nans], axis=0)

        else:
            raise ValueError("Requested trials cannot be less than the number of response trials.")

    def matches(self, other):
        """
        Parameters
        ----------
        other : Trials
            another Trials object

        Returns
        -------
        bool
            whether trial identifiers and response lengths match
        """
        return [*self.index, self.iloc[0].size] == [*other.index, other.iloc[0].size]


def concatenate(*trials, burnin=0):
    """
    Parameters
    ----------
    *trials : Trials
        trial responses to concatenate
    burnin : int
        number of initial frames to discard

    Returns
    -------
    2D array
        [trials, samples]
    """
    # max response trials
    size = max(map(len, trials))

    # convert responses to arrays, filling missing trials with NaNs
    arrays = [r.to_array(size)[:, burnin:] for r in trials]

    # concatenate along samples dimensions
    return np.concatenate(arrays, axis=1)


# ---------------------------- Response Measure ----------------------------

# -- Response Measure Interface --


class Measure:
    """Response Measure"""

    def __call__(self, x):
        """
        Parameters
        ----------
        x : 2D array
            [trials, samples]

        Returns
        -------
        float
            functional measure
        """
        raise NotImplementedError()


# -- Response Measure Types --


class CCMax(Measure):
    """Upper bound of signal correlation -- http://dx.doi.org/10.3389/fncom.2016.00010"""

    def __call__(self, x):
        # number of trials per sample
        trials, samples = x.shape
        t = trials - np.isnan(x).sum(axis=0)

        # pooled variance -> n
        v = 1 / t**2
        w = t - 1
        z = t.sum() - samples
        n = np.sqrt(z / (w * v).sum())

        # response mean
        y_m = np.nanmean(x, axis=0)

        # signal power
        P = np.var(y_m, axis=0, ddof=1)
        TP = np.mean(np.nanvar(x, axis=1, ddof=1), axis=0)
        SP = (n * P - TP) / (n - 1)

        # variance of response mean
        y_m_v = np.var(y_m, axis=0, ddof=0)

        # correlation coefficient ceiling
        return np.sqrt(SP / y_m_v)


# ---------------------------- Response Correlation ----------------------------

# -- Response Correlation Interface --


class Correlation:
    """Response Correlation"""

    def __call__(self, x, y):
        """
        Parameters
        ----------
        x : 2D array
            [trials, samples]
        y : 2D array
            [trials, samples]

        Returns
        -------
        float
            functional measure
        """
        raise NotImplementedError()


# -- Response Correlation Types --


class CCSignal(Correlation):
    """Signal Correlation"""

    def __call__(self, x, y):
        x = np.nanmean(x, axis=0)
        y = np.nanmean(y, axis=0)
        return stats.pearsonr(x, y)[0]
