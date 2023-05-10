import numpy as np
from djutils import merge, rowproperty, rowmethod
from foundation.utils.resample import monotonic
from foundation.virtual import utility, stimulus
from foundation.virtual.bridge import pipe_stim
from foundation.schemas import recording as schema


# -------------- Trial --------------

# -- Trial Base --


class _Trial:
    """Recording Trial"""

    @rowproperty
    def flips(self):
        """
        Returns
        -------
        1D array
            video flip times
        """
        raise NotImplementedError()

    @rowproperty
    def video(self):
        """
        Returns
        -------
        foundation.stimulus.video.Video (virtual)
            tuple
        """
        raise NotImplementedError()


# -- Trial Types --


@schema.lookup
class ScanTrial(_Trial):
    definition = """
    -> pipe_stim.Trial
    """

    @rowproperty
    def flips(self):
        return (pipe_stim.Trial & self).fetch1("flip_times", squeeze=True)

    @rowproperty
    def video(self):
        from foundation.stimulus.video import Video

        trial = pipe_stim.Trial * pipe_stim.Condition & self
        stim_type = trial.fetch1("stimulus_type")
        stim_type = stim_type.split(".")[1]
        return Video.get(stim_type, trial)


# -- Trial --


@schema.link
class Trial:
    links = [ScanTrial]
    name = "trial"
    comment = "trial"


@schema.linkset
class TrialSet:
    link = Trial
    name = "trialset"
    comment = "trial set"


# -- Computed Trial --


@schema.computed
class TrialBounds:
    definition = """
    -> Trial
    ---
    start       : double        # trial start time (seconds)
    end         : double        # trial end time (seconds)
    """

    def make(self, key):
        # trial flip times
        flips = (Trial & key).link.flips

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
    -> Trial
    -> utility.Rate
    ---
    samples     : int unsigned      # number of samples
    """

    def make(self, key):
        from foundation.recording.compute import ResampleTrial

        key["samples"] = (ResampleTrial & key).samples
        self.insert1(key)


@schema.computed
class TrialVideo:
    definition = """
    -> Trial
    ---
    -> stimulus.Video
    """

    def make(self, key):
        key["video_id"] = (Trial & key).link.video.fetch1("video_id")
        self.insert1(key)


# -------------- Trial Filter --------------

# -- Filter Types --


@schema.lookupfilter
class TrialVideoFilter:
    filtertype = Trial
    definition = """
    -> stimulus.VideoFilterSet
    """

    @rowmethod
    def filter(self, trials):
        from foundation.stimulus.video import Video, VideoFilterSet

        # trial videos
        trial_videos = merge(trials, TrialVideo)
        videos = Video & trial_videos

        # filter videos
        videos = (VideoFilterSet & self).filter(videos)

        # filter trials
        return trials & (trial_videos & videos).proj()


# -- Filter --


@schema.filterlink
class TrialFilter:
    links = [TrialVideoFilter]
    name = "trial_filter"
    comment = "trial filter"


@schema.filterlinkset
class TrialFilterSet:
    link = TrialFilter
    name = "trial_filterset"
    comment = "trial filter set"
