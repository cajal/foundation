import io
import av
import numpy as np
from djutils import rowproperty, rowmethod
from foundation.utils import video
from foundation.virtual.bridge import pipe_stim, pipe_gabor, pipe_dot, pipe_rdk
from foundation.schemas import stimulus as schema


# ---------------------------- Video ----------------------------

# -- Video Base --


class _Video:
    """Stimlus Frames"""

    @rowproperty
    def video(self):
        """
        Returns
        -------
        foundation.utils.video.Video
            video object
        """
        raise NotImplementedError()


# -- Video Types --


@schema.lookup
class Clip(_Video):
    definition = """
    -> pipe_stim.Clip
    """

    @rowproperty
    def video(self):
        clip = pipe_stim.Movie * pipe_stim.Movie.Clip * pipe_stim.Clip & self
        clip, start, end, fps = clip.fetch1("clip", "skip_time", "cut_after", "frame_rate")

        start, end = map(float, [start, end])
        start = round(start * fps)
        end = start + round(end * fps)

        frames = []
        reader = av.open(io.BytesIO(clip.tobytes()), mode="r")

        for i, frame in enumerate(reader.decode()):

            if i < start:
                continue
            if i == end:
                break

            frame = frame.to_image().convert(mode="L")
            frames.append(frame)

        return video.Video(frames, period=1 / fps)


@schema.lookup
class Monet2(_Video):
    definition = """
    -> pipe_stim.Monet2
    """

    @rowproperty
    def video(self):
        frames, fps = (pipe_stim.Monet2 & self).fetch1("movie", "fps")
        frames = np.einsum("H W C T -> T H W C", frames)
        return video.Video.fromarray(frames, period=1 / float(fps))


@schema.lookup
class Trippy(_Video):
    definition = """
    -> pipe_stim.Trippy
    """

    @rowproperty
    def video(self):
        frames, fps = (pipe_stim.Trippy & self).fetch1("movie", "fps")
        frames = np.einsum("H W T -> T H W", frames)
        return video.Video.fromarray(frames, period=1 / float(fps))


@schema.lookup
class GaborSequence(_Video):
    definition = """
    -> pipe_stim.GaborSequence
    """

    @rowproperty
    def video(self):
        sequence = (pipe_stim.GaborSequence & self).fetch1()
        fps = (pipe_gabor.Display & sequence).fetch1("fps")

        movs = pipe_gabor.Sequence.Gabor * pipe_gabor.Gabor & sequence
        movs = movs.fetch("movie", order_by="sequence_id ASC")
        assert len(movs) == sequence["sequence_length"]

        return video.Video.fromarray(np.concatenate(movs), period=1 / fps)


@schema.lookup
class DotSequence(_Video):
    definition = """
    -> pipe_stim.DotSequence
    """

    @rowproperty
    def video(self):
        sequence = (pipe_stim.DotSequence & self).fetch1()
        fps = (pipe_dot.Display & sequence).fetch1("fps")

        imgs = pipe_dot.Dot * pipe_dot.Sequence.Dot * pipe_dot.Display & sequence
        imgs = imgs.fetch("image", order_by="dot_id ASC")
        assert len(imgs) == sequence["sequence_length"]

        n_frames, id_trace = (pipe_dot.Trace * pipe_dot.Display & sequence).fetch1("n_frames", "id_trace")
        frames = np.stack(imgs[id_trace])
        assert len(frames) == n_frames

        return video.Video.fromarray(frames, period=1 / fps)


@schema.lookup
class RdkSequence(_Video):
    definition = """
    -> pipe_stim.RdkSequence
    """

    @rowproperty
    def video(self):
        sequence = (pipe_stim.RdkSequence & self).fetch1()
        fps = (pipe_rdk.Display & sequence).fetch1("fps")

        movs = []
        for i in range(sequence["sequence_length"]):
            _key = dict(sequence, sequence_id=i)
            if pipe_rdk.Sequence.Rotation & _key:
                movie = (pipe_rdk.RotationRdk * pipe_rdk.Sequence.Rotation & _key).fetch1("movie")
            elif pipe_rdk.Sequence.Radial & _key:
                movie = (pipe_rdk.RadialRdk * pipe_rdk.Sequence.Radial & _key).fetch1("movie")
            elif pipe_rdk.Sequence.Translation & _key:
                movie = (pipe_rdk.TranslationRdk * pipe_rdk.Sequence.Translation & _key).fetch1("movie")
            else:
                raise Exception
            movs += [movie]

        return video.Video.fromarray(np.concatenate(movs), period=1 / fps)


@schema.lookup
class Frame(_Video):
    definition = """
    -> pipe_stim.Frame
    """

    @rowproperty
    def video(self):
        tup = pipe_stim.StaticImage.Image * pipe_stim.Frame & self
        image, pre_blank, duration = tup.fetch1("image", "pre_blank_period", "presentation_time")
        image = video.Frame.fromarray(image)

        if image.mode == "L":
            blank = np.full([image.height, image.width], 128, dtype=np.uint8)
            blank = video.Frame.fromarray(blank)
        else:
            raise NotImplementedError(f"Frame mode {mode} not implemented")

        if pre_blank > 0:
            return video.Video([blank, image, blank], times=[0, pre_blank, pre_blank + duration])
        else:
            return video.Video([image, blank], [0, duration])


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
        vid = (Video & key).link.video

        key["frames"] = len(vid)
        key["height"] = vid.height
        key["width"] = vid.width
        key["channels"] = vid.channels
        key["mode"] = vid.mode
        key["period"] = vid.period

        self.insert1(key)


# ---------------------------- Filter ----------------------------

# -- Filter Types --


@schema.lookupfilter
class VideoTypeFilter:
    filtertype = Video
    definition = """
    video_type      : varchar(128)  # video type
    include         : bool          # include or exclude
    """

    @rowmethod
    def filter(self, videos):

        if self.fetch1("include"):
            return videos & self
        else:
            return videos - self


@schema.lookupfilter
class VideoSetFilter:
    filtertype = Video
    definition = """
    -> VideoSet
    include         : bool          # include or exclude
    """

    @rowmethod
    def filter(self, videos):

        if self.fetch1("include"):
            return videos & (VideoSet & self).members
        else:
            return videos - (VideoSet & self).members


# -- Filter --


@schema.filterlink
class VideoFilter:
    links = [VideoTypeFilter, VideoSetFilter]
    name = "video_filter"
    comment = "video filter"


@schema.filterlinkset
class VideoFilterSet:
    link = VideoFilter
    name = "video_filterset"
    comment = "video filter set"
