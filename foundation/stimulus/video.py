import numpy as np
from djutils import rowproperty

from foundation.schemas import stimulus as schema
from foundation.utils import video
from foundation.virtual.bridge import pipe_dot, pipe_gabor, pipe_rdk, pipe_stim

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
    ''' Each entry in the table corresponds to a list of frames.
    This table can be used to recreate a video of all frames shown in a scan.
    Example code to fill the table with all frames shown in a scan:

        scan_key = dict(animal_id=26872, session=17, scan_idx=20)
        keys = (pipe_stim.Frame * pipe_stim.Condition * pipe_stim.Trial & scan_key).fetch(
            "KEY", order_by="trial_idx ASC"
        )
        FrameList.fill(
            restrictions=keys,
            note="All stimulus.Frame conditions presented in 26872-17-20, ordered by trial_idx",
        )

    '''
    keys = [pipe_stim.Frame]
    name = "framelist"
    comment = "an ordered list of frames"

    @rowproperty
    def compute(self):
        tups = pipe_stim.StaticImage.Image * pipe_stim.Frame * self.Member() & self
        images = []
        times = []
        current_time = 0
        for image, pre_blank, duration in zip(
            *tups.fetch(
                "image",
                "pre_blank_period",
                "presentation_time",
                order_by="framelist_index",
            )
        ):
            image = video.Frame.fromarray(image)

            if image.mode == "L":
                blank = np.full([image.height, image.width], 128, dtype=np.uint8)
                blank = video.Frame.fromarray(blank)
            else:
                raise NotImplementedError(f"Frame mode {image.mode} not implemented")

            if pre_blank > 0 and current_time == 0:
                images += [blank, image, blank]
                times += [
                    current_time,
                    current_time + pre_blank,
                    current_time + pre_blank + duration,
                ]
            else:
                images += [image, blank]
                times += [current_time + pre_blank, current_time + pre_blank + duration]
            current_time = times[-1]
        return video.Video(images, times=times)


# -- Video --


@schema.link
class Video:
    links = [Clip, Monet2, Trippy, GaborSequence, DotSequence, RdkSequence, Frame]
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
