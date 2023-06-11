from djutils import keys, merge, rowproperty
from foundation.virtual import utility, stimulus, recording


# ----------------------------- Resampling -----------------------------


@keys
class ResampledTrial:
    """Resample Trial"""

    @property
    def key_list(self):
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
            number of resampling time points
        """
        from foundation.utils.resample import samples
        from foundation.utility.resample import Rate

        # trial timing
        start, end = merge(self.key, recording.TrialBounds).fetch1("start", "end")

        # resampling period
        period = (Rate & self.key).link.period

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
        flips = (Trial & self.key).link.flips

        # resampling period
        period = (Rate & self.key).link.period

        # start time
        start = merge(self.key, recording.TrialBounds).fetch1("start")

        # interpolated flip index
        return flip_index(flips - start, period)


# ----------------------------- Scan -----------------------------


@keys
class VisualScanTrials:
    """Visual Scan Trials"""

    @property
    def key_list(self):
        return [
            stimulus.Video,
            recording.ScanRecording,
            recording.TrialFilterSet,
        ]

    @rowproperty
    def trial_ids(self):
        """
        Returns
        -------
        List[str]
            list of trial_ids (foundation.recording.Trial), ordered by trial start
        """
        from foundation.recording.trial import Trial, TrialSet, TrialFilterSet, TrialBounds, TrialVideo

        # all trials
        key = recording.ScanRecording & self.key
        trials = Trial & (TrialSet & key).members

        # filtered trials
        trials = (TrialFilterSet & self.key).filter(trials)

        # video trials
        trials = merge(trials, TrialBounds, TrialVideo) & self.key

        # trial ids, ordered by trial start
        return trials.fetch("trial_id", order_by="start").tolist()
