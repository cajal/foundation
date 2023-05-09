import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm
from djutils import keys, merge, rowproperty, keyproperty, RestrictionError
from foundation.utils.resample import samples, frame_index
from foundation.utility.stat import Summary
from foundation.utility.standardize import Standardize
from foundation.utility.resample import Rate, Offset, Resample
from foundation.stimulus.video import VideoInfo
from foundation.recording.trial import Trial, TrialSet, TrialBounds, TrialVideo
from foundation.recording.trace import Trace, TraceSet, TraceHomogeneous, TraceTrials, TraceSummary


@keys
class ResampleTrial:
    """Resample trial"""

    @property
    def key_list(self):
        return [
            Trial,
            Rate,
        ]

    @rowproperty
    def samples(self):
        # trial timing
        start, end = merge(Trial & self.key, TrialBounds).fetch1("start", "end")

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
        # trial flip times=
        flips = (Trial & self.key).link.flips

        # resampling period
        period = (Rate & self.key).link.period

        # trial and video info
        info = merge(self.key, TrialBounds, TrialVideo, VideoInfo)
        start, frames = info.fetch1("start", "frames")

        if len(flips) != frames:
            raise ValueError("Flips do not match video frames.")

        # sample index for each flip
        index = frame_index(flips - start, period)
        samples = np.arange(index[-1] + 1)

        # first flip of each sampling index
        first = np.diff(index, prepend=-1) > 0

        # for each of the samples, get the previous flip/video index
        previous = interp1d(
            x=index[first],
            y=np.where(first)[0],
            kind="previous",
        )
        return previous(samples).astype(int)


@keys
class TraceResampling:
    """Trace resampling"""

    @property
    def key_list(self):
        return [
            Trace,
            Resample,
            Offset,
            Rate,
        ]

    @rowproperty
    def resample(self):
        """
        Returns
        -------
        foundation.utils.resample.Resample
            callable, resamples traces
        """
        # resampling period, offset, method
        period = (Rate & self.key).link.period
        offset = (Offset & self.key).link.offset
        resample = (Resample & self.key).link.resample

        # trace resampler
        trace = (Trace & self.key).link
        return resample(times=trace.times, values=trace.values, target_period=period, target_offset=offset)


@keys
class ResampleTrace:
    """Resample trace"""

    @property
    def key_list(self):
        return [
            Trace,
            Trial,
            Resample,
            Offset,
            Rate,
        ]

    @keyproperty(Trace, Rate, Offset, Resample)
    def trials(self):
        """
        Returns
        -------
        pandas.Series
            index -- str : trial_id (foundation.recording.trial.Trial)
            data -- 1D array : resampled trace values
        """
        # requested trials
        trials = merge(self.key, TrialBounds)

        # ensure requested trials are valid
        valid_trials = merge(Trace & self.key, TraceTrials)
        if trials - (TrialSet & valid_trials).members:
            raise RestrictionError("Requested trials do not belong to the trace.")

        # fetch trials, ordered by start time
        trial_ids, starts, ends = trials.fetch("trial_id", "start", "end", order_by="start")

        # trace resampler
        resample = (TraceResampling & self.key).resample
        samples = [resample(a, b) for a, b in zip(starts, ends)]

        # pandas Series containing resampled trials
        return pd.Series(
            data=samples,
            index=pd.Index(trial_ids, name="trial_id"),
        )


@keys
class ResampleTraces:
    """Resample trace set"""

    @property
    def key_list(self):
        return [
            TraceSet & "members > 0",
            Trial,
            Resample,
            Offset,
            Rate,
        ]

    @rowproperty
    def trial(self):
        """
        Returns
        ------
        2D array -- [samples, traces]
            resampled traces, ordered by traceset_index
        """
        # trace set
        traces = (TraceSet & self.key).members

        # ensure requested trial is valid
        valid_trials = Trial
        for trial_set in TrialSet & merge(traces, TraceTrials):
            valid_trials &= TrialSet.Member & trial_set

        if self.key - valid_trials:
            raise RestrictionError("Requested trial does not belong to the trace set.")

        # trial start and end times
        start, end = merge(self.key, TrialBounds).fetch1("start", "end")

        # sample traces
        traces = traces.fetch("trace_id", order_by="traceset_index", as_dict=True)
        samples = []
        for trace in tqdm(traces, desc="Traces"):
            sample = (TraceResampling & trace & self.key).resample(start, end)
            samples.append(sample)

        return np.stack(samples, 1)


@keys
class SummarizeTrace:
    """Summarize trace"""

    @property
    def key_list(self):
        return [
            Trace,
            TrialSet & "members > 0",
            Summary,
            Resample,
            Offset,
            Rate,
        ]

    @rowproperty
    def statistic(self):
        """
        Returns
        -------
        float
            trace summary statistic
        """
        # trial set
        trial_keys = (TrialSet & self.key).members

        # resampled trace
        samples = (ResampleTrace & self.key & trial_keys).trials
        samples = np.concatenate(samples)

        # summary statistic
        return (Summary & self.key).link.summary(samples)


@keys
class StandardizeTraces:
    """Trace standardization"""

    @property
    def key_list(self):
        return [
            TraceSet & "members > 0",
            TrialSet & "members > 0",
            Standardize,
            Resample,
            Offset,
            Rate,
        ]

    @rowproperty
    def transform(self):
        """
        Returns
        -------
        foundation.utility.standardize.Standardize
            trace set standardization
        """
        # trace and stat keys
        trace_keys = (TraceSet & self.key).members
        stat_keys = (Standardize & self.key).link.summary_keys

        # homogeneous mask
        hom = merge(trace_keys, TraceHomogeneous)
        hom = hom.fetch("homogeneous", order_by="traceset_index")
        hom = hom.astype(bool)

        # summary stats
        stats = trace_keys * self.key * stat_keys
        stats = merge(stats, TraceSummary)

        # collect stats in dict
        kwargs = dict()
        for skey in stat_keys.proj():
            sid = skey["summary_id"]
            kwargs[sid] = (stats & skey).fetch("summary", order_by="traceset_index")

        # standarization transform
        return (Standardize & self.key).link.standardize(homogeneous=hom, **kwargs)
