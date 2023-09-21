import numpy as np
from djutils import rowproperty

from foundation.virtual.bridge import pipe_stim, pipe_gabor, pipe_dot, pipe_rdk
from foundation.schemas import stimulus as schema

# ---------------------------- Video ----------------------------

# -- Video Interface --


class VideoType:
    """Video Stimulus"""

    @rowproperty
    def compute(self):
        """
        Returns
        -------
        foundation.stimulus.compute.video.VideoType (row)
            compute video
        """
        raise NotImplementedError()


# -- Video Types --


@schema.lookup
class Clip(VideoType):
    definition = """
    -> pipe_stim.Clip
    """

    @rowproperty
    def compute(self):
        from foundation.stimulus.compute.video import Clip

        return Clip & self


@schema.lookup
class Monet2(VideoType):
    definition = """
    -> pipe_stim.Monet2
    """

    @rowproperty
    def compute(self):
        from foundation.stimulus.compute.video import Monet2

        return Monet2 & self


@schema.lookup
class Trippy(VideoType):
    definition = """
    -> pipe_stim.Trippy
    """

    @rowproperty
    def compute(self):
        from foundation.stimulus.compute.video import Trippy

        return Trippy & self


@schema.lookup
class GaborSequence(VideoType):
    definition = """
    -> pipe_stim.GaborSequence
    """

    @rowproperty
    def compute(self):
        from foundation.stimulus.compute.video import GaborSequence

        return GaborSequence & self


@schema.lookup
class DotSequence(VideoType):
    definition = """
    -> pipe_stim.DotSequence
    """

    @rowproperty
    def compute(self):
        from foundation.stimulus.compute.video import DotSequence

        return DotSequence & self


@schema.lookup
class RdkSequence(VideoType):
    definition = """
    -> pipe_stim.RdkSequence
    """

    @rowproperty
    def compute(self):
        from foundation.stimulus.compute.video import RdkSequence

        return RdkSequence & self


@schema.lookup
class Frame(VideoType):
    definition = """
    -> pipe_stim.Frame
    """

    @rowproperty
    def compute(self):
        from foundation.stimulus.compute.video import Frame

        return Frame & self


@schema.list
class FrameList(VideoType):
    """Each entry in the table corresponds to a list of frames.
    This table can be used to recreate a video of all frames shown in a scan.
    Example code to fill the table with all frames shown in a scan and test the result:

    >>> from foundation.stimulus.video import *
    ... import datajoint as dj
    ... scan_key = dict(animal_id=26872, session=17, scan_idx=20)
    ... keys, trial_idx = dj.U('condition_hash').aggr(
    ...     (pipe_stim.Frame * pipe_stim.Condition * pipe_stim.Trial & scan_key),
    ...     trial_idx='MIN(trial_idx)'
    ... ).fetch(
    ...     "KEY", "trial_idx", order_by="trial_idx ASC"
    ... )
    ... frame_list_key = FrameList.fill(
    ...     restrictions=keys,
    ...     note=(
    ...         "All unique stimulus.Frame conditions presented in 26872-17-20, "\
    ...         "ordered by the trial_idx of the first repetition.",
    ...     ),
    ... )
    ... v = (FrameList() & frame_list_key).compute.video
    ... images, preblank, duration = (
    ...     pipe_stim.StaticImage.Image
    ...     * pipe_stim.Trial()
    ...     * pipe_stim.Condition()
    ...     * pipe_stim.Frame()
    ...     & scan_key & f"trial_idx in {tuple(trial_idx)}"
    ... ).fetch("image", "pre_blank_period", "presentation_time", order_by="trial_idx ASC")
    ... for i, j in zip(images, v.frames[1::2]):
    ...     assert np.all(i == j)
    ... v_preblank = np.diff(v.times)[::2]
    ... v_duration = np.diff(v.times)[1::2]
    ... np.allclose(v_preblank, preblank) and np.allclose(v_duration, duration)
    True
    """

    keys = [pipe_stim.Frame]
    name = "framelist"
    comment = "an ordered list of frames"

    @rowproperty
    def compute(self):
        from foundation.stimulus.compute.video import FrameList

        return FrameList & self


# -- Video --


@schema.link
class Video:
    links = [
        Clip,
        Monet2,
        Trippy,
        GaborSequence,
        DotSequence,
        RdkSequence,
        Frame,
        FrameList,
    ]
    name = "video"
    comment = "video stimulus"


@schema.linkset
class VideoSet:
    link = Video
    name = "videoset"
    comment = "video stimulus set"


# -- Computed Video --


@schema.computed
class VideoInfo:
    definition = """
    -> Video
    ---
    frames          : int unsigned  # video frames
    height          : int unsigned  # video height
    width           : int unsigned  # video width
    channels        : int unsigned  # video channels
    mode            : varchar(16)   # video mode
    period=NULL     : double        # video period (seconds)
    """

    def make(self, key):
        vid = (Video & key).link.compute.video

        key["frames"] = len(vid)
        key["height"] = vid.height
        key["width"] = vid.width
        key["channels"] = vid.channels
        key["mode"] = vid.mode
        key["period"] = vid.period

        self.insert1(key)
