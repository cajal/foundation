import numpy as np
import pandas as pd
from tqdm import tqdm
from djutils import keys, merge, rowproperty, keyproperty, RestrictionError
from foundation.virtual import utility, stimulus, recording


# ----------------------------- Resampling -----------------------------


@keys
class ResampleTrial:
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
    def video_index(self):
        """
        Returns
        -------
        1D array
            video frame index for each of the resampled time points
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


@keys
class TraceResampling:
    """Trace Resampling"""

    @property
    def key_list(self):
        return [
            recording.Trace,
            utility.Resample,
            utility.Offset,
            utility.Rate,
        ]

    @rowproperty
    def resample(self):
        """
        Returns
        -------
        foundation.utils.resample.Resample
            callable, resamples traces
        """
        from foundation.utility.resample import Rate, Offset, Resample
        from foundation.recording.trace import Trace

        # resampling period, offset, method
        period = (Rate & self.key).link.period
        offset = (Offset & self.key).link.offset
        resample = (Resample & self.key).link.resample

        # trace resampler
        trace = (Trace & self.key).link
        return resample(times=trace.times, values=trace.values, target_period=period, target_offset=offset)


@keys
class ResampleTrace:
    """Resample Trace"""

    @property
    def key_list(self):
        return [
            recording.Trace,
            recording.Trial,
            utility.Resample,
            utility.Offset,
            utility.Rate,
        ]

    @keyproperty(recording.Trace, utility.Rate, utility.Offset, utility.Resample)
    def trials(self):
        """
        Returns
        -------
        pandas.Series
            index -- str : trial_id (foundation.recording.trial.Trial)
            data -- 1D array : resampled trace values
        """
        from foundation.recording.trial import TrialSet

        # requested trials
        trials = merge(self.key, recording.TrialBounds)

        # ensure requested trials are valid
        valid_trials = merge(recording.Trace & self.key, recording.TraceTrials)
        if trials - (TrialSet & valid_trials).members:
            raise RestrictionError("Requested trials do not belong to the trace.")

        # fetch trials, ordered by start time
        trial_ids, starts, ends = trials.fetch("trial_id", "start", "end", order_by="start")

        # resample traces
        resample = (TraceResampling & self.key).resample
        samples = [resample(a, b) for a, b in zip(starts, ends)]

        # pandas Series containing resampled trials
        return pd.Series(
            data=samples,
            index=pd.Index(trial_ids, name="trial_id"),
        )


@keys
class ResampleTraces:
    """Resample Trace Set"""

    @property
    def key_list(self):
        return [
            recording.TraceSet & "members > 0",
            recording.Trial,
            utility.Resample,
            utility.Offset,
            utility.Rate,
        ]

    @rowproperty
    def trial(self):
        """
        Returns
        ------
        2D array -- [samples, traces]
            resampled traces, ordered by traceset_index
        """
        from foundation.recording.trace import TraceSet

        # trace set
        traces = (TraceSet & self.key).members

        # ensure requested trial is valid
        valid_trials = recording.Trial
        for trial_set in recording.TrialSet & merge(traces, recording.TraceTrials):
            valid_trials &= recording.TrialSet.Member & trial_set

        if self.key - valid_trials:
            raise RestrictionError("Requested trial does not belong to the trace set.")

        # trial start and end times
        start, end = merge(self.key, recording.TrialBounds).fetch1("start", "end")

        # sample traces
        traces = traces.fetch("trace_id", order_by="traceset_index", as_dict=True)
        samples = []
        for trace in tqdm(traces, desc="Traces"):
            sample = (TraceResampling & trace & self.key).resample(start, end)
            samples.append(sample)

        return np.stack(samples, 1)


# ----------------------------- Statistics -----------------------------


@keys
class SummarizeTrace:
    """Summarize Trace"""

    @property
    def key_list(self):
        return [
            recording.Trace,
            recording.TrialSet & "members > 0",
            utility.Summary,
            utility.Resample,
            utility.Offset,
            utility.Rate,
        ]

    @rowproperty
    def statistic(self):
        """
        Returns
        -------
        float
            trace summary statistic
        """
        from foundation.utility.stat import Summary
        from foundation.recording.trial import TrialSet

        # trial set
        trial_keys = (TrialSet & self.key).members

        # resampled trace
        samples = (ResampleTrace & self.key & trial_keys).trials
        samples = np.concatenate(samples)

        # summary statistic
        return (Summary & self.key).link.summary(samples)


# ----------------------------- Standardization -----------------------------


@keys
class StandardizeTraces:
    """Trace Standardization"""

    @property
    def key_list(self):
        return [
            recording.TraceSet & "members > 0",
            recording.TrialSet & "members > 0",
            utility.Standardize,
            utility.Resample,
            utility.Offset,
            utility.Rate,
        ]

    @rowproperty
    def transform(self):
        """
        Returns
        -------
        foundation.utility.standardize.Standardize
            trace set standardization
        """
        from foundation.utility.standardize import Standardize
        from foundation.recording.trace import TraceSet

        # standardization link
        stand = (Standardize & self.key).link

        # trace and stat keys
        trace_keys = (TraceSet & self.key).members
        stat_keys = stand.summary_keys

        # homogeneous mask
        hom = merge(trace_keys, recording.TraceHomogeneous)
        hom = hom.fetch("homogeneous", order_by="traceset_index")
        hom = hom.astype(bool)

        # summary stats
        stats = trace_keys * self.key * stat_keys
        stats = merge(stats, recording.TraceSummary)

        # collect stats in dict
        kwargs = dict()
        for skey in stat_keys.proj():
            sid = skey["summary_id"]
            kwargs[sid] = (stats & skey).fetch("summary", order_by="traceset_index")

        # standarization transform
        return stand.standardize(homogeneous=hom, **kwargs)


# ----------------------------- Response -----------------------------


@keys
class VisualResponse:
    """Visual Response"""

    @property
    def key_list(self):
        return [
            recording.Trace,
            utility.Resample,
            utility.Offset,
            utility.Rate,
            stimulus.Video,
        ]

    @rowproperty
    def trials(self):
        """
        Returns
        -------
        pandas.Series
            index -- str -- trial_id (foundation.recording.trial.Trial)
            data -- 1D array -- [samples], trial response
        """
        from foundation.recording.trial import TrialSet

        trials = merge(self.key, recording.TraceTrials)
        trials = (TrialSet & trials).members

        key = merge(trials, recording.TrialVideo) * self.key
        return (ResampleTrace & key).trials

    @rowproperty
    def mean(self):
        """
        Returns
        -------
        1D array -- [samples]
            mean response
        """
        from foundation.utils.resample import truncate

        trials = truncate(*self.trials)
        return np.stack(trials, 0).mean(0)

    @rowproperty
    def timing(self):
        """
        Returns
        -------
        float
            response period (seconds)
        float
            response offset (seconds)
        """
        # TODO
        raise NotImplementedError()
