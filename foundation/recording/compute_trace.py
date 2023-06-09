import numpy as np
import pandas as pd
from djutils import keys, merge, rowproperty, keyproperty, RestrictionError
from foundation.utils import tqdm
from foundation.virtual import utility, stimulus, recording


# ----------------------------- Trace/Traces -----------------------------


@keys
class Trace:
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
    def resampler(self):
        """
        Returns
        -------
        foundation.utils.resample.Resample
            callable that resamples traces
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
class Traces:
    """Trace Set Resampling"""

    @property
    def key_list(self):
        return [
            recording.TraceSet & "members > 0",
            utility.Resample,
            utility.Offset,
            utility.Rate,
        ]

    @rowproperty
    def resamplers(self):
        """
        Returns
        -------
        tuple[foundation.utils.resample.Resample]
            tuple of callable that resamples traces, ordered by traceset_index
        """
        from foundation.recording.trace import TraceSet

        traces = (TraceSet & self.key).members
        traces = traces.fetch("trace_id", order_by="traceset_index", as_dict=True)
        traces = tqdm(traces, desc="Traces")

        resamplers = []
        for trace in traces:
            resampler = (Trace & trace & self.key).resampler
            resamplers.append(resampler)

        return tuple(resamplers)


# ----------------------------- Resample Trace/Traces -----------------------------


@keys
class ResampledTrace:
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
            index -- str
                : trial_id (foundation.recording.trial.Trial)
            data -- 1D array
                : [timepoints] ; resampled trace values
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
        resampler = (Trace & self.key).resampler
        samples = [resampler(a, b) for a, b in zip(starts, ends)]

        # pandas Series containing resampled trials
        return pd.Series(data=samples, index=pd.Index(trial_ids, name="trial_id"))


@keys
class ResampledTraces:
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
        from foundation.recording.trial import TrialSet
        from foundation.recording.trace import TraceSet

        # ensure requested trial is valid
        trial_sets = TrialSet & merge((TraceSet & self.key).members, recording.TraceTrials)
        for trial_set in trial_sets.proj():
            if self.key - (TrialSet & trial_set).members:
                raise RestrictionError("Requested trial does not belong to the trace set.")

        # trial start and end times
        start, end = merge(self.key, recording.TrialBounds).fetch1("start", "end")

        # resample traces
        samples = [r(start, end) for r in (Traces & self.key).resamplers]

        # [samples, traces]
        return np.stack(samples, 1)


# ----------------------------- Trace Statistics -----------------------------


@keys
class TraceSummary:
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
        samples = (ResampledTrace & self.key & trial_keys).trials
        samples = np.concatenate(samples)

        # summary statistic
        return (Summary & self.key).link.summary(samples)


# ----------------------------- Standardized Trace/Traces -----------------------------


@keys
class StandardizedTrace:
    """Standardized Trace"""

    @property
    def key_list(self):
        return [
            recording.Trace,
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
            callable, standardizes trace
        """
        from foundation.utility.standardize import Standardize

        # standardization link
        stand = (Standardize & self.key).link

        # stat keys
        stat_keys = stand.summary_keys

        # homogeneous mask
        hom = merge(self.key, recording.TraceHomogeneous)
        hom = hom.fetch1("homogeneous")
        hom = [hom.astype(bool)]

        # summary stats
        stats = self.key * stat_keys
        stats = merge(stats, recording.TraceSummary)

        # stats dict
        summary_id, summary = stats.fetch("summary_id", "summary")
        kwargs = {k: [v] for k, v in zip(summary_id, summary)}

        # standarization transform
        return stand.standardize(homogeneous=hom, **kwargs)


@keys
class StandardizedTraces:
    """Standardized Trace Set"""

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
            callable, standardizes trace set
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


# ----------------------------- Visual Trace -----------------------------


@keys
class VisualTrace:
    """Visual Trace"""

    @property
    def key_list(self):
        return [
            stimulus.Video,
            recording.TrialFilterSet,
            recording.Trace,
            recording.TrialSet,
            utility.Standardize,
            utility.Resample,
            utility.Offset,
            utility.Rate,
        ]

    @rowproperty
    def response(self):
        """
        Returns
        -------
        foundation.utils.response.Trials
            response trials
        """
        from foundation.utils.response import Trials
        from foundation.recording.trial import Trial, TrialVideo, TrialSet, TrialFilterSet

        # all trials
        trials = merge(recording.Trace & self.key, recording.TraceTrials)
        trials = Trial & (TrialSet & trials).members

        # restricted trials
        trials = (TrialFilterSet & self.key).filter(trials)
        trials = merge(trials, TrialVideo) & self.key

        # trace trials
        trials = (ResampledTrace & trials & self.key).trials

        # trace standardization
        transform = (StandardizedTrace & self.key).transform

        # visual response
        return Trials(data=map(transform, trials.values), index=trials.index, tolerance=1)
