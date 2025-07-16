import numpy as np
import datajoint as dj
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
class Gratezk(VideoType):
    definition = """
    -> pipe_stim.Gratezk
    """

    @rowproperty
    def compute(self):
        from foundation.stimulus.compute.video import Gratezk

        return Gratezk & self

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
    keys = [pipe_stim.Frame]
    name = "framelist"
    comment = "an ordered list of frames"

    @rowproperty
    def compute(self):
        from foundation.stimulus.compute.video import FrameList

        return FrameList & self


@schema.list
class Frame2List(VideoType):
    keys = [pipe_stim.Frame2]
    name = "frame2list"
    comment = "an ordered list of frame2 stimuli"

    @rowproperty
    def compute(self):
        from foundation.stimulus.compute.video import Frame2List

        return Frame2List & self
    

# -- Video --


@schema.link
class Video:
    links = [
        Clip,
        Monet2,
        Gratezk,
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
