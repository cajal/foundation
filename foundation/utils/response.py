import numpy as np
import pandas as pd
from .resample import truncate


class Response(pd.Series):
    """Response Trials"""

    def __init__(self, data, index=None, tolerance=0):
        """
        Parameters
        ----------
        data : Sequence[1D array]
            trial responses
        index : Sequence[str] | None
            trial identifiers. optional if len(data) == 1
        tolerance : 1
            response length mismatch tolerance
        """
        # truncate trial responses to the same length
        data = truncate(*data, tolerance=tolerance)

        # assert all responses are 1D
        assert all(map(lambda x: x.ndim == 1, data)), "Each response must be 1D."

        # initialize
        super().__init__(data=data, index=[None] if index is None else index)

    def to_array(self, trials=None):
        """
        Parameters
        ----------
        trials : int | None
            number of trials

        Returns
        -------
        2D array
            [trials, samples], nan-filled to desired trial size
        """
        # 2D response array
        array = np.stack(self.values, axis=0)

        if trials is None or trials == len(self):
            return array

        elif trials > len(self):
            # nans for missing trials
            _trials = trials - len(self)
            samples = self.iloc[0].size
            dtype = self.iloc[0].dtype
            nans = np.full([_trials, samples], np.nan, dtype=dtype)

            # nan-filled array
            return np.concatenate([array, nans], axis=0)

        else:
            raise ValueError("Requested trials cannot be less than the number of response trials.")

    def matches(self, other):
        """
        Parameters
        ----------
        other : Response
            another response object

        Returns
        -------
        bool
            whether trial identifiers and response lengths match
        """
        return [*self.index, self.iloc[0].size] == [*other.index, other.iloc[0].size]


def concatenate(responses):
    """
    Parameters
    ----------
    Sequence[Response]
        trial responses to concatenate

    Returns
    -------
    2D array
        [trials, samples]
    """
    # max response trials
    trials = max(map(len, responses))

    # convert responses to arrays, filling missing trials with NaNs
    arrays = [r.to_array(trials) for r in responses]

    # concatenate along samples dimensions
    return np.concatenate(arrays, axis=1)
