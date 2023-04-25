import numpy as np
import datajoint as dj
from djutils import link, row_property, skip_missing
from foundation.bridge.pipeline import pipe_stim
from foundation.stimulus import video
from foundation.recording import resample

schema = dj.schema("foundation_recording")


# -------------- Trial --------------

# -- Trial Base --


class TrialBase:
    """Recording Trial"""

    @row_property
    def bounds(self):
        """
        Returns
        -------
        float
            trial start time (seconds)
        float
            trial end time (seconds)
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
    def bounds(self):
        flips = self.flips
        return self.flips[0], self.flips[-1]

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
class TrialBounds(dj.Computed):
    definition = """
    -> TrialLink
    ---
    start       : double        # trial start time (seconds)
    end         : double        # trial end time (seconds)
    """

    @skip_missing
    def make(self, key):
        key["start"], key["end"] = (TrialLink & key).link.bounds
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
        start, end = (TrialBounds & key).fetch1("start", "end")
        period = (resample.RateLink & key).link.period
        key["samples"] = round((end - start) / period) + 1
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
        from foundation.utils.trace import monotonic

        # flip times and sampling period
        flips = (TrialLink & key).link.flips
        period = (resample.RateLink & key).link.period

        assert np.isfinite(flips).all()
        assert monotonic(flips)

        # video and trial info
        info = video.VideoInfo * TrialVideo * TrialSamples * TrialBounds & key
        start, samples, fixed, frames = info.fetch1("start", "samples", "fixed", "frames")

        assert frames == len(flips)

        # nearest flip if fixed, else previous flip
        index = interp1d(
            x=(flips - start) / period,
            y=np.arange(flips.size),
            kind="nearest" if fixed else "previous",
            fill_value="extrapolate",
            bounds_error=False,
        )

        key["video_index"] = index(np.arange(samples)).astype(int)
        self.insert1(key)


# @schema
# class TrialVideo(dj.Computed):
#     definition = """
#     -> TrialLink
#     ---
#     -> video.VideoLink
#     flips           : int unsigned      # number of video flips
#     flip_start      : double            # time of first flip
#     flip_end        : double            # time of last flip
#     """

#     def make(self, key):
#         from foundation.utils.trace import monotonic

#         try:
#             trial_link = (TrialLink & key).link
#             video = trial_link.video
#             flips = trial_link.flips

#         except MissingError:
#             logger.warning(f"Missing data. Not populating {key}")

#         assert np.isfinite(flips).all()
#         assert monotonic(flips)

#         key["flips"] = len(flips)
#         key["flip_start"] = flips[0]
#         key["flip_end"] = flips[-1]
#         key["video_id"] = video.fetch1("video_id")

#         self.insert1(key)


# @schema
# class TrialResample(dj.Computed):
#     definition = """
#     -> TrialVideo
#     -> resample.RateLink
#     ---
#     samples         : int unsigned      # number of samples
#     video_index     : longblob          # video frame index for each sample
#     """

#     @property
#     def key_source(self):
#         key = TrialVideo * video.VideoInfo & "frames = flips"
#         return TrialVideo.proj() * resample.RateLink.proj() & key

#     def make(self, key):
#         from scipy.interpolate import interp1d

#         try:
#             flips = (TrialLink & key).link.flips
#             period = (resample.RateLink & key).link.period

#         except MissingError:
#             logger.warning(f"Missing data. Not populating {key}")
#             return

#         # start and end flip times, fixed frame rate
#         info = video.VideoInfo * TrialVideo & key
#         start, end, fixed = info.fetch1("flip_start", "flip_end", "fixed")

#         # nearest flip if fixed, else previous flip
#         index = interp1d(
#             x=(flips - start) / period,
#             y=np.arange(flips.size),
#             kind="nearest" if fixed else "previous",
#             fill_value="extrapolate",
#             bounds_error=False,
#         )

#         # interpolate samples
#         key["samples"] = round((end - start) / period) + 1
#         key["video_index"] = index(np.arange(key["samples"])).astype(int)

#         self.insert1(key)


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
