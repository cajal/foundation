import numpy as np
from djutils import merge, row_property, row_method
from foundation.utils.trace import monotonic, samples, frame_index
from foundation.utility import resample
from foundation.stimulus import video
from foundation.schemas.pipeline import pipe_stim
from foundation.schemas import recording as schema


# -------------- Trial --------------

# -- Trial Base --


class _Trial:
    """Recording Trial"""

    @row_property
    def flips(self):
        """
        Returns
        -------
        1D array
            video flip times
        """
        raise NotImplementedError()

    @row_property
    def video(self):
        """
        Returns
        -------
        video.VideoLink
            tuple from video.VideoLink
        """
        raise NotImplementedError()


# -- Trial Types --


@schema.lookup
class ScanTrial(_Trial):
    definition = """
    -> pipe_stim.Trial
    """

    @row_property
    def flips(self):
        return (pipe_stim.Trial & self).fetch1("flip_times", squeeze=True)

    @row_property
    def video(self):
        trial = pipe_stim.Trial * pipe_stim.Condition & self
        stim_type = trial.fetch1("stimulus_type")
        stim_type = stim_type.split(".")[1]
        return video.VideoLink.get(stim_type, trial)


# -- Trial Link --


@schema.link
class TrialLink:
    links = [ScanTrial]
    name = "trial"
    comment = "recording trial"


@schema.set
class TrialSet:
    keys = [TrialLink]
    name = "trials"
    comment = "set of recording trials"


# -- Computed Trial --


@schema.computed
class TrialBounds:
    definition = """
    -> TrialLink
    ---
    start       : double        # trial start time (seconds)
    end         : double        # trial end time (seconds)
    """

    def make(self, key):
        # trial flip times
        flips = (TrialLink & key).link.flips

        # verify flip times
        assert np.isfinite(flips).all()
        assert monotonic(flips)

        # trial bounds
        key["start"] = flips[0]
        key["end"] = flips[-1]
        self.insert1(key)


@schema.computed
class TrialSamples:
    definition = """
    -> TrialBounds
    -> resample.RateLink
    ---
    samples     : int unsigned  # number of trial samples
    """

    def make(self, key):
        # trial timing
        start, end = (TrialBounds & key).fetch1("start", "end")
        period = (resample.RateLink & key).link.period

        # trial samples
        key["samples"] = samples(start, end, period)
        self.insert1(key)


@schema.computed
class TrialVideo:
    definition = """
    -> TrialLink
    ---
    -> video.VideoLink
    """

    def make(self, key):
        key["video_id"] = (TrialLink & key).link.video.fetch1("video_id")
        self.insert1(key)


@schema.computed
class VideoSamples:
    definition = """
    -> TrialVideo
    -> TrialSamples
    ---
    video_index     : longblob      # video frame index for each sample
    """

    def make(self, key):
        from scipy.interpolate import interp1d

        # flip times and sampling period
        flips = (TrialLink & key).link.flips
        period = (resample.RateLink & key).link.period

        # ensure flips and video frames match
        frames = merge(TrialVideo & key, video.VideoInfo).fetch1("frames")
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


# -------------- Trial Filter --------------

# -- Trial Filter Base --


class _TrialFilter:
    """Trial Filter"""

    @row_method
    def filter(self, trials):
        """
        Parameters
        ----------
        trials : TrialLink
            TrialLink tuples

        Returns
        -------
        TrialLink
            retricted TrialLink tuples
        """
        raise NotImplementedError()


# -- Trial Filter Types --


@schema.lookup
class TrialVideoFilter(_TrialFilter):
    definition = """
    -> video.VideoFilterSet
    """

    @row_method
    def filter(self, trials):
        # trial videos
        trial_videos = merge(trials, TrialVideo)
        videos = video.VideoLink & trial_videos

        # filter videos
        for key in (video.VideoFilterSet & self).members.fetch("KEY", order_by="member_id"):
            videos = (video.VideoFilterLink & key).link.filter(videos)

        return trials & (trial_videos & videos).proj()


# -- Trial Filter Link --


@schema.link
class TrialFilterLink:
    links = [TrialVideoFilter]
    name = "trial_filter"
    comment = "recording trial filter"


@schema.set
class TrialFilterSet:
    keys = [TrialFilterLink]
    name = "trial_filters"
    comment = "set of recording trial filters"
