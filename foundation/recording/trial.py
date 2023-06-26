from djutils import merge, rowproperty, rowmethod
from foundation.virtual import utility, stimulus, scan
from foundation.virtual.bridge import pipe_stim, pipe_shared
from foundation.schemas import recording as schema


# ---------------------------- Trial ----------------------------

# -- Trial Base --


class TrialType:
    """Recording Trial"""

    @rowproperty
    def compute(self):
        """
        Returns
        -------
        foundation.recording.compute_trial.TrialType (row)
            compute trial
        """
        raise NotImplementedError()


# -- Trial Types --


@schema.lookup
class ScanTrial(TrialType):
    definition = """
    -> pipe_stim.Trial
    """

    @rowproperty
    def compute(self):
        from foundation.recording.compute_trial import ScanTrial

        return ScanTrial & self


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
class TrialVideo:
    definition = """
    -> Trial
    ---
    -> stimulus.Video
    """

    def make(self, key):
        key["video_id"] = (Trial & key).link.compute.video_id
        self.insert1(key)


@schema.computed
class TrialBounds:
    definition = """
    -> Trial
    ---
    start       : double        # trial start time (seconds)
    end         : double        # trial end time (seconds)
    """

    def make(self, key):
        key["start"], key["end"] = (Trial & key).link.compute.bounds
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


# ---------------------------- Trial Filter ----------------------------

# -- Trial Filter Types --


@schema.lookupfilter
class VideoTypeFilter:
    filtertype = Trial
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
class VideoSetFilter:
    filtertype = Trial
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
class ScanPupilFilter:
    filtertype = Trial
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
