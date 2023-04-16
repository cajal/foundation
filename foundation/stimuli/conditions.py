import numpy as np
import datajoint as dj
import io
import av
from PIL import Image
from tqdm import tqdm
from foundation.utils.logging import logger

stimulus = dj.create_virtual_module("stimulus", "pipeline_stimulus")
gabor = dj.create_virtual_module("gabor", "pipeline_gabor")
dot = dj.create_virtual_module("dot", "pipeline_dot")
rdk = dj.create_virtual_module("rdk", "pipeline_rdk")

schema = dj.schema("foundation_stimuli")


class FramesMixin:
    @property
    def frames(self):
        """
        Returns
        -------
        List[Image]
            stimulus frames
        """
        raise NotImplementedError


@schema
class Clip(dj.Computed, FramesMixin):
    definition = """
    -> stimulus.Clip
    ---
    frames      : int unsigned  # number of stimulus frames
    """

    @staticmethod
    def decode(condition_hash):
        key = dict(condition_hash=condition_hash)

        clip = stimulus.Movie * stimulus.Movie.Clip * stimulus.Clip & key
        clip, start, end, fps = clip.fetch1("clip", "skip_time", "cut_after", "frame_rate")

        start, end = map(float, [start, end])
        start = round(start * fps)
        end = start + round(end * fps)

        reader = av.open(io.BytesIO(clip.tobytes()), mode="r")
        for i, frame in enumerate(reader.decode()):

            if i < start:
                continue
            if i == end:
                return

            yield frame.to_image().convert(mode="L")

    def make(self, key):
        frames = 0
        for _ in self.decode(**key):
            frames += 1

        key["frames"] = frames
        self.insert1(key)

    @property
    def frames(self):
        condition_hash, n = self.fetch1("condition_hash", "frames")
        frames = list(self.decode(condition_hash))
        assert len(frames) == n
        return frames


@schema
class Monet2(dj.Computed, FramesMixin):
    definition = """
    -> stimulus.Monet2
    ---
    frames      : int unsigned  # number of stimulus frames
    """

    def make(self, key):
        movie = (stimulus.Monet2 & key).fetch1("movie")

        key["frames"] = movie.shape[-1]
        self.insert1(key)

    @property
    def frames(self):
        key, n = self.fetch1(dj.key, "frames")
        movie = (stimulus.Monet2 & key).fetch1("movie").squeeze(2)
        return [Image.fromarray(movie[..., i]) for i in range(movie.shape[-1])]


@schema
class Trippy(dj.Computed, FramesMixin):
    definition = """
    -> stimulus.Trippy
    ---
    frames      : int unsigned  # number of stimulus frames
    """

    def make(self, key):
        movie = (stimulus.Trippy & key).fetch1("movie")

        key["frames"] = movie.shape[-1]
        self.insert1(key)

    @property
    def frames(self):
        key, n = self.fetch1(dj.key, "frames")
        movie = (stimulus.Trippy & key).fetch1("movie")
        return [Image.fromarray(movie[..., i]) for i in range(movie.shape[-1])]


@schema
class GaborSequence(dj.Computed, FramesMixin):
    definition = """
    -> stimulus.GaborSequence
    ---
    frames      : int unsigned  # number of stimulus frames
    """

    def make(self, key):
        sequence = (stimulus.GaborSequence & key).fetch1()

        n_frames = (gabor.Sequence.Gabor * gabor.Gabor & sequence).fetch("n_frames")
        assert len(n_frames) == sequence["sequence_length"]

        key["frames"] = sum(n_frames)
        self.insert1(key)

    @property
    def frames(self):
        sequence = (stimulus.GaborSequence * self).fetch1()

        movs = (gabor.Sequence.Gabor * gabor.Gabor & sequence).fetch("movie", order_by="sequence_id")
        assert len(movs) == sequence["sequence_length"]

        frames = np.concatenate(movs)
        assert len(frames) == sequence["frames"]

        return [Image.fromarray(frame, mode="L") for frame in frames]


@schema
class DotSequence(dj.Computed, FramesMixin):
    definition = """
    -> stimulus.DotSequence
    ---
    frames      : int unsigned  # number of stimulus frames
    """

    def make(self, key):
        sequence = (stimulus.DotSequence & key).fetch1()

        key["frames"] = (dot.Trace * dot.Display & sequence).fetch1("n_frames")
        self.insert1(key)

    @property
    def frames(self):
        sequence = (stimulus.DotSequence * self).fetch1()

        images = (dot.Dot * dot.Sequence.Dot * dot.Display & sequence).fetch("image", order_by="dot_id")
        assert len(images) == sequence["sequence_length"]

        n_frames, id_trace = (dot.Trace * dot.Display & sequence).fetch1("n_frames", "id_trace")
        frames = np.stack(images[id_trace])
        assert len(frames) == n_frames

        return [Image.fromarray(frame, mode="L") for frame in frames]


@schema
class RdkSequence(dj.Computed, FramesMixin):
    definition = """
    -> stimulus.RdkSequence
    ---
    frames      : int unsigned  # number of stimulus frames
    """

    def make(self, key):
        sequence = (stimulus.RdkSequence & key).fetch1()

        keys = [dict(sequence, sequence_id=i) for i in range(sequence["sequence_length"])]
        tables = [
            rdk.RotationRdk * rdk.Sequence.Rotation,
            rdk.RadialRdk * rdk.Sequence.Radial,
            rdk.TranslationRdk * rdk.Sequence.Translation,
        ]
        frames = np.concatenate([(table & keys).fetch("n_frames") for table in tables])
        assert len(frames) == sequence["sequence_length"]

        key["frames"] = frames.sum()
        self.insert1(key)

    @property
    def frames(self):
        sequence = (stimulus.RdkSequence * self).fetch1()

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
        frames = np.concatenate(movs)

        return [Image.fromarray(frame, mode="L") for frame in frames]
