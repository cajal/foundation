import numpy as np
import datajoint as dj
from djutils import link, group, merge, row_property, row_method, skip_missing
from foundation.utils.trace import monotonic, samples, frame_index
from foundation.utility import resample
from foundation.stimulus import video
from foundation.schemas.pipeline import pipe_stim
from foundation.schemas import recording as schema


# -------------- Trial --------------

# -- Trial Base --


class TrialBase:
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


@schema
class ScanTrial(TrialBase, dj.Lookup):
    definition = """
    -> pipe_stim.Trial
    """

    @row_property
    def flips(self):
        return (pipe_stim.Trial & self).fetch1("flip_times", squeeze=True)

    @row_property
    def bounds(self):
        flips = self.flips
        return self.flips[0], self.flips[-1]

    @row_property
    def video(self):
        trial = pipe_stim.Trial * pipe_stim.Condition & self
        stim_type = trial.fetch1("stimulus_type")
        stim_type = stim_type.split(".")[1]
        return video.VideoLink.get(stim_type, trial)


# -- Trial Link --


@link(schema)
class TrialLink:
    links = [ScanTrial]
    name = "trial"
    comment = "recording trial"


@group(schema)
class TrialSet:
    keys = [TrialLink]
    name = "trials"
    comment = "set of recording trials"


# -- Computed Trial --


@schema
class TrialBounds(dj.Computed):
    definition = """
    -> TrialLink
    ---
    start       : double        # trial start time (seconds)
    end         : double        # trial end time (seconds)
    """

    @skip_missing
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


@schema
class TrialSamples(dj.Computed):
    definition = """
    -> TrialBounds
    -> resample.RateLink
    ---
    samples     : int unsigned  # number of trial samples
    """

    @skip_missing
    def make(self, key):
        # trial timing
        start, end = (TrialBounds & key).fetch1("start", "end")
        period = (resample.RateLink & key).link.period

        # trial samples
        key["samples"] = samples(start, end, period)
        self.insert1(key)


@schema
class TrialVideo(dj.Computed):
    definition = """
    -> TrialLink
    ---
    -> video.VideoLink
    """

    @skip_missing
    def make(self, key):
        key["video_id"] = (TrialLink & key).link.video.fetch1("video_id")
        self.insert1(key)


@schema
class VideoSamples(dj.Computed):
    definition = """
    -> TrialVideo
    -> TrialSamples
    ---
    video_index     : longblob      # video frame index for each sample
    """

    @skip_missing
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


class TrialFilterBase:
    """Trial Filter"""

    @row_method
    def filter(self, trials):
        """
        Parameters
        ----------
        trials : TrialLink
            Trial tuples

        Returns
        -------
        TrialLink
            retricted Trial tuples
        """
        raise NotImplementedError()


# -- Trial Filter Types --


@schema
class TrialVideoFilter(TrialFilterBase, dj.Lookup):
    definition = """
    -> video.VideoFilterSet
    """

    @row_method
    def filter(self, trials):
        # trial videos
        trial_videos = merge(trials, TrialVideo)
        videos = video.VideoLink & trial_videos

        # filter videos
        for key in (video.VideoFilterSet & self).members.fetch(dj.key, order_by="member_id"):
            videos = (video.VideoFilterLink & key).link.filter(videos)

        return trials & (trial_videos & videos).proj()


# -- Trial Filter Link --


@link(schema)
class TrialFilterLink:
    links = [TrialVideoFilter]
    name = "trial_filter"
    comment = "recording trial filter"


@group(schema)
class TrialFilterSet:
    keys = [TrialFilterLink]
    name = "trial_filters"
    comment = "set of recording trial filters"
