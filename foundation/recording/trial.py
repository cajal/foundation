import numpy as np
from scipy.interpolate import interp1d
from djutils import merge, row_property, row_method
from foundation.utils.resample import monotonic, frame_index
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
        foundation.stimulus.video.VideoLink
            tuple
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

    @row_method
    def resampled_video_index(self, rate_link):
        """
        Parameters
        ----------
        rate_link : foundation.utility.resample.RateLink
            tuple

        Returns
        -------
        1D array
            video frame index
            dtype = int
        """
        # flip times
        flips = self.link.flips

        # trial and video info
        info = merge(self, TrialBounds, TrialVideo, video.VideoInfo)
        start, frames = info.fetch1("start", "frames")

        if len(flips) != frames:
            raise ValueError("Flips do not match video frames.")

        # resampling period
        period = rate_link.link.period

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

        # video frame index for each sample
        return previous(samples).astype(int)


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
class TrialVideo:
    definition = """
    -> TrialLink
    ---
    -> video.VideoLink
    """

    def make(self, key):
        key["video_id"] = (TrialLink & key).link.video.fetch1("video_id")
        self.insert1(key)


# -------------- Trial Filter --------------


@schema.filter_lookup
class TrialVideoFilter:
    filter_type = TrialLink
    definition = """
    -> video.VideoFilterSet
    """

    @row_method
    def filter(self, trials):
        # trial videos
        trial_videos = merge(trials, TrialVideo)
        videos = video.VideoLink & trial_videos

        # filter videos
        videos = (video.VideoFilterSet & self).filter(videos)

        return trials & (trial_videos & videos).proj()


@schema.filter_link
class TrialFilterLink:
    filters = [TrialVideoFilter]
    name = "trial_filter"
    comment = "recording trial filter"


@schema.filter_link_set
class TrialFilterSet:
    filter_link = TrialFilterLink
    name = "trial_filters"
    comment = "set of recording trial filters"
