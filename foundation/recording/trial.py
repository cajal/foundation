import numpy as np
from djutils import merge, rowproperty, rowmethod
from foundation.utils.resample import monotonic
from foundation.virtual import utility, stimulus, scan
from foundation.virtual.bridge import pipe_stim, pipe_shared
from foundation.schemas import recording as schema


# ---------------------------- Trial ----------------------------

# -- Trial Base --


class _Trial:
    """Recording Trial"""

    @rowproperty
    def flips(self):
        """
        Returns
        -------
        1D array
            stimulus flip times
        """
        raise NotImplementedError()

    @rowproperty
    def video_id(self):
        """
        Returns
        -------
        str
            video_id (foundation.stimulus.video.Video)
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
    def video_id(self):
        from foundation.stimulus.video import Video

        trial = pipe_stim.Trial * pipe_stim.Condition & self
        stim_type = trial.fetch1("stimulus_type")
        stim_type = stim_type.split(".")[1]
        return Video.get(stim_type, trial).fetch1("video_id")


# -- Trial --


@schema.link
class Trial:
    links = [ScanTrial]
    name = "trial"
    comment = "recording trial"


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
        from foundation.recording.compute_trial import ResampledTrial

        key["samples"] = (ResampledTrial & key).samples
        self.insert1(key)


@schema.computed
class TrialVideo:
    definition = """
    -> Trial
    ---
    -> stimulus.Video
    """

    def make(self, key):
        key["video_id"] = (Trial & key).link.video_id
        self.insert1(key)


# ---------------------------- Trial Filter ----------------------------


# -- Trial Filter Base --


class _TrialFilter:
    filtertype = Trial


# -- Trial Filter Types --


@schema.lookupfilter
class VideoTypeFilter(_TrialFilter):
    definition = """
    video_type      : varchar(128)  # video type
    include         : bool          # include or exclude
    """

    @rowmethod
    def filter(self, trials):
        key = merge(trials, TrialVideo, stimulus.Video) & self

        if self.fetch1("include"):
            return trials & key.proj()
        else:
            return trials - key.proj()


@schema.lookupfilter
class VideoSetFilter(_TrialFilter):
    definition = """
    -> stimulus.VideoSet
    include         : bool          # include or exclude
    """

    @rowmethod
    def filter(self, trials):
        from foundation.stimulus.video import VideoSet

        key = merge(trials, TrialVideo, stimulus.Video) & (VideoSet & self).members

        if self.fetch1("include"):
            return trials & key.proj()
        else:
            return trials - key.proj()


@schema.lookupfilter
class ScanPupilFilter(_TrialFilter):
    definition = """
    -> pipe_shared.TrackingMethod
    max_nans        : decimal(4, 3) # maximum tolerated fraction of nans
    """

    @rowmethod
    def filter(self, trials):
        key = merge(trials, self, Trial.ScanTrial, scan.PupilNans) & "nans < max_nans"

        return trials & key.proj()


# -- Trial Filter --


@schema.filterlink
class TrialFilter:
    links = [VideoTypeFilter, VideoSetFilter, ScanPupilFilter]
    name = "trial_filter"
    comment = "trial filter"


@schema.filterlinkset
class TrialFilterSet:
    link = TrialFilter
    name = "trial_filterset"
    comment = "trial filter set"
