import numpy as np
from djutils import keys, merge, rowmethod, rowproperty
from foundation.utils import tqdm
from foundation.virtual import utility, stimulus, recording


# ----------------------------- Trace/Traces -----------------------------


@keys
class Trace:
    """Trace"""

    @property
    def key_list(self):
        return [
            recording.Trace,
        ]

    @rowproperty
    def valid_trials(self):
        """
        Returns
        -------
        foundation.recording.Trial (rows)
            valid trials
        """
        from foundation.recording.trial import Trial, TrialSet

        # trace trials
        key = merge(self.key, recording.TraceTrials)
        return Trial & (TrialSet & key).members


@keys
class Traces:
    """Trace Set"""

    @property
    def key_list(self):
        return [
            recording.TraceSet & "members > 0",
        ]

    @rowproperty
    def valid_trials(self):
        """
        Returns
        -------
        foundation.recording.Trial (rows)
            valid trials
        """
        from foundation.recording.trace import TraceSet
        from foundation.recording.trial import Trial, TrialSet

        # trace set trials
        key = (TraceSet & self.key).members
        key = merge(key, recording.TraceTrials)
        return Trial & (TrialSet & key).members


# ----------------------------- Resample Trace/Traces -----------------------------


@keys
class ResampledTrace:
    """Resampled Trace"""

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
            callable, trace resampler
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

    @rowmethod
    def trial(self, trial_id):
        """
        Parameters
        ----------
        trial_id : str
            key (foundation.recording.trial.Trial)

        Returns
        -------
        1D array -- [samples]
            resampled trace
        """
        # recording trial
        trial = recording.Trial.proj() & {"trial_id": trial_id}

        # ensure trial is valid
        assert not trial - (Trace & self.key).valid_trials, "Invalid trial"

        # trial start and end times
        start, end = merge(trial, recording.TrialBounds).fetch1("start", "end")

        # resampled trace
        return self.resampler(start, end)

    @rowmethod
    def trials(self, trial_ids):
        """
        Parameters
        ----------
        trial_id : Sequence[str]
            sequence of keys (foundation.recording.trial.Trial)

        Yields
        ------
        1D array -- [samples]
            resampled trace
        """
        # recording trials
        trials = recording.Trial.proj() & [dict(trial_id=trial_id) for trial_id in trial_ids]

        # ensure trials are valid
        assert not trials - (Trace & self.key).valid_trials, "Invalid trials"

        # trial start and end times
        starts, ends = merge(trials, recording.TrialBounds).fetch("start", "end")

        # trace resampler
        resampler = self.resampler

        for start, end in zip(starts, ends):
            # resampled trace
            yield resampler(start, end)


@keys
class ResampledTraces:
    """Resampled Trace Set"""

    @property
    def key_list(self):
        return [
            recording.TraceSet,
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
            tuple of callables, trace resamplers, ordered by traceset_index
        """
        from foundation.recording.trace import TraceSet

        # trace set
        traces = (TraceSet & self.key).members
        traces = traces.fetch("trace_id", order_by="traceset_index", as_dict=True)
        traces = tqdm(traces, desc="Traces")

        # trace resamplers
        resamplers = []
        for trace in traces:
            resampler = (ResampledTrace & trace & self.key).resampler
            resamplers.append(resampler)

        return tuple(resamplers)

    @rowmethod
    def trial(self, trial_id):
        """
        Parameters
        ----------
        trial_id : str
            key (foundation.recording.trial.Trial)

        Returns
        -------
        2D array -- [samples, traces]
            resampled traces, ordered by traceset index
        """
        # recording trial
        trial = recording.Trial.proj() & {"trial_id": trial_id}

        # ensure trial is valid
        assert not trial - (Traces & self.key).valid_trials, "Invalid trial"

        # trial start and end times
        start, end = merge(trial, recording.TrialBounds).fetch1("start", "end")

        # resampled traces
        return np.stack([r(start, end) for r in self.resamplers], axis=1)


# ----------------------------- Trace Statistics -----------------------------


@keys
class TraceSummary:
    """Trace Summary"""

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

        # recording trials
        trial_ids = (TrialSet & self.key).members.fetch("trial_id", order_by="trial_id")

        # resampled traces
        trials = (ResampledTrace & self.key).trials(trial_ids)
        trials = np.concatenate(list(trials))

        # summary statistic
        return (Summary & self.key).link.summary(trials)


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
        stat_keys = stand.summaries

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
        stat_keys = stand.summaries

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
