import numpy as np
import datajoint as dj
from djutils import link, group, merge, row_property, skip_missing
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
        from foundation.utils.trace import monotonic

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
        # trial duration
        start, end = (TrialBounds & key).fetch1("start", "end")
        duration = end - start

        # trial samples
        key["samples"] = (resample.RateLink & key).link.samples(duration)
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

        # video and trial info
        info = video.VideoInfo * TrialVideo * TrialSamples * TrialBounds & key
        start, samples, fixed, frames = info.fetch1("start", "samples", "fixed", "frames")

        # ensure flips and video frames match
        assert frames == len(flips)

        # nearest flip if fixed, else previous flip
        index = interp1d(
            x=(flips - start) / period,
            y=np.arange(flips.size),
            kind="nearest" if fixed else "previous",
            fill_value="extrapolate",
            bounds_error=False,
        )

        # video frame index
        key["video_index"] = index(np.arange(samples)).astype(int)
        self.insert1(key)


# -------------- Trial Filter --------------

# -- Trial Filter Base --


# class TrialFilterBase:
#     """Trial Filter"""

#     @row_method
#     def filter(self, trials):
#         """
#         Parameters
#         ----------
#         trials : Trial
#             Trial tuples

#         Returns
#         -------
#         Trial
#             retricted Trial tuples
#         """
#         raise NotImplementedError()


# # -- Trial Filter Types --


# @method(schema)
# class FlipsEqualsFrames(TrialFilterBase):
#     name = "flips_equals_frames"
#     comment = "flips == frames"

#     @row_method
#     def filter(self, trials):
#         key = (trials * stimulus.Stimulus) & "flips = frames"
#         return trials & key.proj()


# @schema
# class StimulusType(TrialFilterBase, dj.Lookup):
#     definition = """
#     stimulus_type       : varchar(128)  # stimulus type
#     """

#     def filter(self, trials):
#         return trials & (stimulus.StimulusLink & self)


# # -- Trial Filter Link --


# @link(schema)
# class TrialFilterLink:
#     links = [FlipsEqualsFrames, StimulusType]
#     name = "trial_filter"
#     comment = "recording trial filter"


# # -- Trial Filter Group --


# @group(schema)
# class TrialFilterGroup:
#     keys = [TrialFilterLink]
#     name = "trial_filters"
#     comment = "recording trial filters"
