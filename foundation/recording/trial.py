import numpy as np
import datajoint as dj
from djutils import link, group, method, row_method, row_property, MissingError
from foundation.utils.logging import logger
from foundation.bridge.pipeline import pipe_stim
from foundation.stimulus import video
from foundation.recording import resample

schema = dj.schema("foundation_recording")


# -------------- Trial --------------

# -- Trial Base --


class TrialBase:
    """Recording Trial"""

    @row_property
    def video(self):
        """
        Returns
        -------
        video.VideoLink
            tuple from video.VideoLink
        """
        raise NotImplementedError()

    @row_property
    def flips(self):
        """
        Returns
        -------
        1D array
            video flip times
        """
        raise NotImplementedError()


# -- Trial Types --


@schema
class ScanTrial(TrialBase, dj.Lookup):
    definition = """
    -> pipe_stim.Trial
    """

    @row_property
    def video(self):
        trial = pipe_stim.Trial * pipe_stim.Condition & self
        stim_type = trial.fetch1("stimulus_type")
        stim_type = stim_type.split(".")[1]
        return video.VideoLink.get(stim_type, trial)

    @row_property
    def flips(self):
        return (pipe_stim.Trial & self).fetch1("flip_times", squeeze=True)


# -- Trial Link --


@link(schema)
class TrialLink:
    links = [ScanTrial]
    name = "trial"
    comment = "recording trial"


# -- Computed Trial --


@schema
class TrialVideo(dj.Computed):
    definition = """
    -> TrialLink
    ---
    -> video.VideoLink
    """

    def make(self, key):
        try:
            video = (TrialLink & key).link.video

        except MissingError:
            logger.warning(f"Missing video. Not populating {key}")

        key["video_id"] = video.fetch1("video_id")
        self.insert1(key)


@schema
class TrialFlips(dj.Computed):
    definition = """
    -> TrialLink
    ---
    flips       : int unsigned      # number of video flips
    flip_start  : double            # time of first flip
    flip_end    : double            # time of last flip
    """

    def make(self, key):
        from foundation.utils.trace import monotonic

        try:
            flips = (TrialLink & key).link.flips

        except MissingError:
            logger.warning(f"Missing flips. Not populating {key}")

        assert np.isfinite(flips).all()
        assert monotonic(flips)

        key["flips"] = len(flips)
        key["flip_start"] = flips[0]
        key["flip_end"] = flips[-1]

        self.insert1(key)


@schema
class TrialSamples(dj.Computed):
    definition = """
    -> TrialLink
    -> resample.RateLink
    ---
    samples         : int unsigned      # number of samples
    video_index     : longblob          # video frame index for each sample
    """

    @property
    def key_source(self):
        key = TrialFlips * TrialVideo * video.VideoFrames & "frames = flips"
        return TrialFlips.proj() * resample.RateLink.proj() & key

    def make(self, key):
        from scipy.interpolate import interp1d

        try:
            flips = (TrialLink & key).link.flips
            period = (resample.RateLink & key).link.period

        except MissingError:
            logger.warning(f"Missing data. Not populating {key}")
            return

        start = flips[0]
        end = flips[-1]
        key["samples"] = round((end - start) / period) + 1

        index = interp1d(
            x=flips - start,
            y=np.arange(flips.size),
            kind="nearest",
            fill_value="extrapolate",
            bounds_error=False,
        )
        key["video_index"] = index(np.arange(key["samples"]) * period).astype(int)

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
