from djutils import keys, merge, rowproperty
from foundation.virtual import utility, recording


# ----------------------------- Resample -----------------------------


@keys
class ResampledTrial:
    """Resample Trial"""

    @property
    def keys(self):
        return [
            recording.Trial,
            utility.Rate,
        ]

    @rowproperty
    def samples(self):
        """
        Returns
        -------
        int
            number of resampled time points
        """
        from foundation.utils.resample import samples
        from foundation.utility.resample import Rate

        # trial timing
        start, end = merge(self.key, recording.TrialBounds).fetch1("start", "end")

        # resampling period
        period = (Rate & self.item).link.period

        # trial samples
        return samples(start, end, period)

    @rowproperty
    def flip_index(self):
        """
        Returns
        -------
        1D array
            stimulus flip index for each of the resampled time points
        """
        from foundation.utils.resample import flip_index
        from foundation.utility.resample import Rate
        from foundation.recording.trial import Trial

        # trial flip times
        flips = (Trial & self.item).link.compute.flip_times

        # resampling period
        period = (Rate & self.item).link.period

        # start time
        start = merge(self.key, recording.TrialBounds).fetch1("start")

        # interpolated flip index
        return flip_index(flips - start, period)
