import io
import av
import numpy as np
import datajoint as dj
from djutils import link, group, row_property, row_method
from PIL import Image
from foundation.utils.video import Video
from foundation.schemas.pipeline import pipe_stim
from foundation.schemas import stimulus as schema


# -------------- Video --------------

# -- Video Base --


class VideoBase:
    """Stimlus Frames"""

    @row_property
    def video(self):
        """
        Returns
        -------
        Video
        """
        raise NotImplementedError()


# -- Video Types --


@schema
class Clip(VideoBase, dj.Lookup):
    definition = """
    -> pipe_stim.Clip
    """

    @row_property
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

        return Video(frames)


@schema
class Monet2(VideoBase, dj.Lookup):
    definition = """
    -> pipe_stim.Monet2
    """

    @row_property
    def video(self):
        movie = (pipe_stim.Monet2 & self).fetch1("movie").squeeze(2)
        return Video([Image.fromarray(movie[..., i]) for i in range(movie.shape[-1])])


@schema
class Trippy(VideoBase, dj.Lookup):
    definition = """
    -> pipe_stim.Trippy
    """

    @row_property
    def video(self):
        movie = (stimulus.Trippy & self).fetch1("movie")
        return Video([Image.fromarray(movie[..., i]) for i in range(movie.shape[-1])])


@schema
class GaborSequence(VideoBase, dj.Lookup):
    definition = """
    -> pipe_stim.GaborSequence
    """

    @row_property
    def video(self):
        gabor = dj.create_virtual_module("gabor", "pipeline_gabor")
        sequence = (pipe_stim.GaborSequence & self).fetch1()

        movs = (gabor.Sequence.Gabor * gabor.Gabor & sequence).fetch("movie", order_by="sequence_id")
        assert len(movs) == sequence["sequence_length"]

        return Video([Image.fromarray(frame, mode="L") for frame in np.concatenate(movs)])


@schema
class DotSequence(VideoBase, dj.Lookup):
    definition = """
    -> pipe_stim.DotSequence
    """

    @row_property
    def video(self):
        dot = dj.create_virtual_module("dot", "pipeline_dot")
        sequence = (pipe_stim.DotSequence & self).fetch1()

        images = (dot.Dot * dot.Sequence.Dot * dot.Display & sequence).fetch("image", order_by="dot_id")
        assert len(images) == sequence["sequence_length"]

        n_frames, id_trace = (dot.Trace * dot.Display & sequence).fetch1("n_frames", "id_trace")
        frames = np.stack(images[id_trace])
        assert len(frames) == n_frames

        return Video([Image.fromarray(frame, mode="L") for frame in frames])


@schema
class RdkSequence(VideoBase, dj.Lookup):
    definition = """
    -> pipe_stim.RdkSequence
    """

    @row_property
    def video(self):
        rdk = dj.create_virtual_module("rdk", "pipeline_rdk")
        sequence = (pipe_stim.RdkSequence & self).fetch1()

        movs = []
        for i in range(sequence["sequence_length"]):
            _key = dict(sequence, sequence_id=i)
            if rdk.Sequence.Rotation & _key:
                movie = (rdk.RotationRdk * rdk.Sequence.Rotation & _key).fetch1("movie")
            elif rdk.Sequence.Radial & _key:
                movie = (rdk.RadialRdk * rdk.Sequence.Radial & _key).fetch1("movie")
            elif rdk.Sequence.Translation & _key:
                movie = (rdk.TranslationRdk * rdk.Sequence.Translation & _key).fetch1("movie")
            else:
                raise Exception
            movs += [movie]

        return Video([Image.fromarray(frame, mode="L") for frame in np.concatenate(movs)])


@schema
class Frame(VideoBase, dj.Lookup):
    definition = """
    -> pipe_stim.Frame
    """

    @row_property
    def video(self):
        tup = pipe_stim.StaticImage.Image * pipe_stim.Frame & self
        image, pre_blank = tup.fetch1("image", "pre_blank_period")
        image = Image.fromarray(image)

        if image.mode == "L":
            blank = np.full([image.height, image.width], 128, dtype=np.uint8)
            blank = Image.fromarray(blank)
        else:
            raise NotImplementedError(f"Image mode {mode} not implemented")

        if pre_blank > 0:
            return Video([blank, image, blank], fixed=False)
        else:
            return Video([image, blank], fixed=False)


# -- Video Link --


@link(schema)
class VideoLink:
    links = [Clip, Monet2, Trippy, GaborSequence, DotSequence, RdkSequence, Frame]
    name = "video"
    comment = "video stimulus"


@group(schema)
class VideoSet:
    keys = [VideoLink]
    name = "videos"
    comment = "set of video stimuli"


# -- Computed Video --


@schema
class VideoInfo(dj.Computed):
    definition = """
    -> VideoLink
    ---
    frames      : int unsigned  # video frames
    height      : int unsigned  # video height
    width       : int unsigned  # video width
    channels    : int unsigned  # video channels
    mode        : varchar(16)   # video mode
    fixed       : bool          # fixed frame rate
    """

    def make(self, key):
        frames = (VideoLink & key).link.video

        key["frames"] = len(frames)
        key["height"] = frames.height
        key["width"] = frames.width
        key["channels"] = frames.channels
        key["mode"] = frames.mode
        key["fixed"] = frames.fixed

        self.insert1(key)


# -------------- Video Filter --------------

# -- Video Filter Base --


class VideoFilterBase:
    """Video Filter"""

    @row_method
    def filter(self, videos):
        """
        Parameters
        ----------
        videos : VideoLink
            VideoLink tuples

        Returns
        -------
        VideoLink
            retricted VideoLink tuples
        """
        raise NotImplementedError()


# -- Video Filter Types --


@schema
class VideoTypeFilter(VideoFilterBase, dj.Lookup):
    definition = """
    video_type      : varchar(128)      # video type
    include         : bool              # include or exclude
    """

    @row_method
    def filter(self, videos):

        if self.fetch1("include"):
            return videos & self
        else:
            return videos - self


@schema
class VideoSetFilter(VideoFilterBase, dj.Lookup):
    definition = """
    -> VideoSet
    include         : bool              # include or exclude
    """

    @row_method
    def filter(self, videos):

        if self.fetch1("include"):
            return videos & (VideoSet & self).members
        else:
            return videos - (VideoSet & self).members


# -- Video Filter Link --


@link(schema)
class VideoFilterLink:
    links = [VideoTypeFilter, VideoSetFilter]
    name = "video_filter"
    comment = "video filter"


@group(schema)
class VideoFilterSet:
    keys = [VideoFilterLink]
    name = "video_filters"
    comment = "set of video filters"
