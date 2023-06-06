import numpy as np


class Measure:
    """Functional Measure"""

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
