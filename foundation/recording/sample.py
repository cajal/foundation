import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from djutils import merge, row_property
from foundation.utils.trace import samples, frame_index
from foundation.utility import resample
from foundation.stimulus import video
from foundation.recording import trial, trace
from foundation.schemas import recording as schema


@schema.computed
class TrialSamples:
    definition = """
    -> trial.TrialBounds
    -> resample.RateLink
    ---
    samples     : int unsigned  # number of trial samples
    """

    def make(self, key):
        # trial timing
        start, end = (trial.TrialBounds & key).fetch1("start", "end")
        period = (resample.RateLink & key).link.period

        # trial samples
        key["samples"] = samples(start, end, period)
        self.insert1(key)


@schema.computed
class VideoSamples:
    definition = """
    -> trial.TrialVideo
    -> TrialSamples
    ---
    video_index     : longblob      # video frame index for each sample
    """

    def make(self, key):
        # flip times and sampling period
        flips = (trial.TrialLink & key).link.flips
        period = (resample.RateLink & key).link.period

        # ensure flips and video frames match
        frames = merge(trial.TrialVideo & key, video.VideoInfo).fetch1("frames")
        assert len(flips) == frames

        # sampling index for each flip
        samp_index = frame_index(flips - flips[0], period)
        new_samp = np.diff(samp_index, prepend=-1) > 0

        # samples
        samps = np.arange(samp_index[-1] + 1)
        assert len(samps) == (TrialSamples & key).fetch1("samples")

        # for each of the samples, get the previous flip/video index
        previous = interp1d(
            x=samp_index[new_samp],
            y=np.where(new_samp)[0],
            kind="previous",
        )
        key["video_index"] = previous(samps).astype(int)
        self.insert1(key)


@schema.computed
class TraceSamples:
    definition = """
    -> trace.TraceTrials
    -> resample.RateLink
    -> resample.OffsetLink
    -> resample.ResampleLink
    ---
    trace           : longblob          # resampled trace
    samples         : int unsigned      # number of samples
    nans            : int unsigned      # number of nans
    """

    def make(self, key):
        # resampling
        period = (resample.RateLink & key).link.period
        offset = (resample.OffsetLink & key).link.offset
        resampler = (resample.ResampleLink & key).link.resampler

        # trace resampler
        trace_link = (trace.TraceLink & key).link
        r = resampler(times=trace_link.times, values=trace_link.values, target_period=period)

        # trials
        trials = merge((trace.TraceTrials & key).trials.members, trial.TrialBounds)

        # sample trials, ordered by member_id
        start, end = trials.fetch("start", "end", order_by="member_id")
        samps = [r(s, e, offset) for s, e in zip(start, end)]
        samps = np.concatenate(samps).astype(np.float32)

        # insert key
        self.insert1(dict(key, trace=samps, samples=len(samps), nans=np.isnan(samps).sum()))

    @row_property
    def samples(self):
        """
        Returns
        -------
        pd.DataFrame
            index -- trial_id
            trace -- resampled trace
        """
        # fetch data
        key, samps, total = self.fetch1("KEY", "trace", "samples")

        # trials
        trials = merge((trace.TraceTrials & key).trials.members, TrialSamples & key)

        # trial samples, ordered by member_id
        trial_id, trial_samples = trials.fetch("trial_id", "samples", order_by="member_id")

        # trace split indices
        *split, _total = np.cumsum(trial_samples)
        assert _total == total == len(samps)

        # dataframe containing trial samples
        return pd.DataFrame(
            data={"trace": np.split(samps, split)},
            index=pd.Index(trial_id, name="trial_id"),
        )
