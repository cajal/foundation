import io
import av
import numpy as np
import datajoint as dj
from djutils import link
from PIL import Image
from foundation.utils.video import Video


schema = dj.schema("foundation_stimuli")


# ---------- Stimulus Base ----------


class StimulusBase:
    _definition = """
    {key}
    ---
    frames      : int unsigned  # number of frames
    height      : int unsigned  # frame height
    width       : int unsigned  # frame height
    channels    : int unsigned  # number of channels
    mode        : varchar(16)   # stimulus mode
    """

    @staticmethod
    def _frames(**key):
        """
        Parameters
        ----------
        key : dict
            primary key

        Returns
        -------
        Video
        """
        raise NotImplementedError()

    @property
    def frames(self):
        """
        Returns
        -------
        Video
        """
        key, T, H, W, C, mode = self.fetch1(dj.key, "frames", "height", "width", "channels", "mode")
        frames = self._frames(**key)

        assert len(frames) == T
        assert frames.height == H
        assert frames.width == W
        assert frames.channels == C
        assert frames.mode == mode

        return frames

    def make(self, key):
        frames = self._frames(**key)
        _key = {
            "frames": len(frames),
            "height": frames.height,
            "width": frames.width,
            "channels": frames.channels,
            "mode": frames.mode,
        }
        self.insert1(dict(**key, **_key))


# ---------- Stimulus Types ----------

stimulus = dj.create_virtual_module("stimulus", "pipeline_stimulus")
gabor = dj.create_virtual_module("gabor", "pipeline_gabor")
dot = dj.create_virtual_module("dot", "pipeline_dot")
rdk = dj.create_virtual_module("rdk", "pipeline_rdk")


@schema
class Clip(StimulusBase, dj.Computed):
    definition = StimulusBase._definition.format(key="-> stimulus.Clip")

    @staticmethod
    def _frames(**key):
        clip = stimulus.Movie * stimulus.Movie.Clip * stimulus.Clip & key
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

            frame = frame.to_image().convert(mode="L")  # TODO: currently assumes all clips are grayscale
            frames.append(frame)

        return Video(frames)


@schema
class Monet2(StimulusBase, dj.Computed):
    definition = StimulusBase._definition.format(key="-> stimulus.Monet2")

    @staticmethod
    def _frames(**key):
        movie = (stimulus.Monet2 & key).fetch1("movie").squeeze(2)
        return Video([Image.fromarray(movie[..., i]) for i in range(movie.shape[-1])])


@schema
class Trippy(StimulusBase, dj.Computed):
    definition = StimulusBase._definition.format(key="-> stimulus.Trippy")

    @staticmethod
    def _frames(**key):
        movie = (stimulus.Trippy & key).fetch1("movie")
        return Video([Image.fromarray(movie[..., i]) for i in range(movie.shape[-1])])


@schema
class GaborSequence(StimulusBase, dj.Computed):
    definition = StimulusBase._definition.format(key="-> stimulus.GaborSequence")

    @staticmethod
    def _frames(**key):
        sequence = (stimulus.GaborSequence & key).fetch1()

        movs = (gabor.Sequence.Gabor * gabor.Gabor & sequence).fetch("movie", order_by="sequence_id")
        assert len(movs) == sequence["sequence_length"]

        return Video([Image.fromarray(frame, mode="L") for frame in np.concatenate(movs)])


@schema
class DotSequence(StimulusBase, dj.Computed):
    definition = StimulusBase._definition.format(key="-> stimulus.DotSequence")

    @staticmethod
    def _frames(**key):
        sequence = (stimulus.DotSequence & key).fetch1()

        images = (dot.Dot * dot.Sequence.Dot * dot.Display & sequence).fetch("image", order_by="dot_id")
        assert len(images) == sequence["sequence_length"]

        n_frames, id_trace = (dot.Trace * dot.Display & sequence).fetch1("n_frames", "id_trace")
        frames = np.stack(images[id_trace])
        assert len(frames) == n_frames

        return Video([Image.fromarray(frame, mode="L") for frame in frames])


@schema
class RdkSequence(StimulusBase, dj.Computed):
    definition = StimulusBase._definition.format(key="-> stimulus.RdkSequence")

    @staticmethod
    def _frames(**key):
        sequence = (stimulus.RdkSequence & key).fetch1()

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
class Frame(StimulusBase, dj.Computed):
    definition = StimulusBase._definition.format(key="-> stimulus.Frame")

    @staticmethod
    def _frames(**key):
        tup = stimulus.StaticImage.Image * stimulus.Frame & key
        image, pre_blank = tup.fetch1("image", "pre_blank_period")
        image = Image.fromarray(image)

        if image.mode == "L":
            blank = np.full([image.height, image.width], 128, dtype=np.uint8)
            blank = Image.fromarray(blank)
        else:
            raise NotImplementedError(f"Image mode {mode} not implemented")

        if pre_blank > 0:
            return Video([blank, image, blank])
        else:
            return Video([image, blank])


# ---------- Stimulus Link ----------


@link(schema)
class Stimulus:
    links = [Clip, Monet2, Trippy, GaborSequence, DotSequence, RdkSequence, Frame]
    name = "stimulus"
    comment = "stimulus frames"
