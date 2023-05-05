import numpy as np
from djutils import merge, row_property, row_method
from foundation.utils.resample import monotonic
from foundation.virtual.bridge import pipe_stim
from foundation.virtual import stimulus
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
        foundation.stimulus.video.VideoLink (virtual)
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
        return stimulus.VideoLink.get(stim_type, trial)


# -- Trial --


@schema.link
class TrialLink:
    links = [ScanTrial]
    name = "trial"
    comment = "trial"


# -- Trial Set --


@schema.link_set
class TrialSet:
    link = TrialLink
    name = "trials"
    comment = "trial set"


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
    -> stimulus.VideoLink
    """

    def make(self, key):
        key["video_id"] = (TrialLink & key).link.video.fetch1("video_id")
        self.insert1(key)


# -------------- Trial Filter --------------

# -- Filter Types --


@schema.filter_lookup
class TrialVideoFilter:
    ftype = TrialLink
    definition = """
    -> stimulus.VideoFilterSet
    """

    @row_method
    def filter(self, trials):
        from foundation.stimulus.video import VideoLink, VideoFilterSet

        # trial videos
        trial_videos = merge(trials, TrialVideo)
        videos = VideoLink & trial_videos

        # filter videos
        videos = (VideoFilterSet & self).filter(videos)

        # filter trials
        return trials & (trial_videos & videos).proj()


# -- Filter --


@schema.filter_link
class TrialFilterLink:
    links = [TrialVideoFilter]
    name = "trial_filter"
    comment = "trial filter"


# -- Filter Set --


@schema.filter_link_set
class TrialFilterSet:
    link = TrialFilterLink
    name = "trial_filters"
    comment = "trial filter set"
