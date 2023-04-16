import numpy as np
import datajoint as dj
import io
import av
from PIL import Image
from foundation.utils.video import Video


stimulus = dj.create_virtual_module("stimulus", "pipeline_stimulus")
gabor = dj.create_virtual_module("gabor", "pipeline_gabor")
dot = dj.create_virtual_module("dot", "pipeline_dot")
rdk = dj.create_virtual_module("rdk", "pipeline_rdk")

schema = dj.schema("foundation_stimuli")


# ---------- Condition Mixin ----------


class ConditionMixin:
    definition = """
    -> stimulus.{table}
    ---
    frames      : int unsigned  # number of stimulus frames
    """

    @staticmethod
    def _frames(condition_hash):
        """
        Parameters
        ----------
        condition_hash : str
            primary key of stimulus.Condition

        Returns
        -------
        int
            number of stimulus frames
        """
        raise NotImplementedError()

    @property
    def frames(self):
        """
        Returns
        -------
        Video
            stimulus frames
        """
        raise NotImplementedError

    def make(self, key):
        key["frames"] = self._frames(**key)
        self.insert1(key)


# ---------- stimulus.Clip ----------


@schema
class Clip(ConditionMixin, dj.Computed):
    definition = ConditionMixin.definition.format(table="Clip")

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

            yield frame.to_image().convert(mode="L")  # TODO: currently assumes that all clips are grayscale

    @staticmethod
    def _frames(condition_hash):
        frames = 0
        for _ in Clip.decode(condition_hash):
            frames += 1

        return frames

    @property
    def frames(self):
        condition_hash, n = self.fetch1("condition_hash", "frames")
        frames = Video(self.decode(condition_hash))
        assert len(frames) == n
        return frames


# ---------- stimulus.Monet2 ----------


@schema
class Monet2(ConditionMixin, dj.Computed):
    definition = ConditionMixin.definition.format(table="Monet2")

    @staticmethod
    def _frames(condition_hash):
        key = dict(condition_hash=condition_hash)
        return (stimulus.Monet2 & key).fetch1("movie").shape[-1]

    @property
    def frames(self):
        key, n = self.fetch1(dj.key, "frames")
        movie = (stimulus.Monet2 & key).fetch1("movie").squeeze(2)

        frames = Video([Image.fromarray(movie[..., i]) for i in range(movie.shape[-1])])
        assert len(frames) == n

        return frames


# ---------- stimulus.Trippy ----------


@schema
class Trippy(ConditionMixin, dj.Computed):
    definition = ConditionMixin.definition.format(table="Trippy")

    @staticmethod
    def _frames(condition_hash):
        key = dict(condition_hash=condition_hash)
        return (stimulus.Trippy & key).fetch1("movie").shape[-1]

    @property
    def frames(self):
        key, n = self.fetch1(dj.key, "frames")
        movie = (stimulus.Trippy & key).fetch1("movie")

        frames = Video([Image.fromarray(movie[..., i]) for i in range(movie.shape[-1])])
        assert len(frames) == n

        return frames


# ---------- stimulus.GaborSequence ----------


@schema
class GaborSequence(ConditionMixin, dj.Computed):
    definition = ConditionMixin.definition.format(table="GaborSequence")

    @staticmethod
    def _frames(condition_hash):
        key = dict(condition_hash=condition_hash)
        sequence = (stimulus.GaborSequence & key).fetch1()

        n_frames = (gabor.Sequence.Gabor * gabor.Gabor & sequence).fetch("n_frames")
        assert len(n_frames) == sequence["sequence_length"]

        return sum(n_frames)

    @property
    def frames(self):
        sequence = (stimulus.GaborSequence * self).fetch1()

        movs = (gabor.Sequence.Gabor * gabor.Gabor & sequence).fetch("movie", order_by="sequence_id")
        assert len(movs) == sequence["sequence_length"]

        frames = np.concatenate(movs)
        assert len(frames) == sequence["frames"]

        return Video([Image.fromarray(frame, mode="L") for frame in frames])


# ---------- stimulus.DotSequence ----------


@schema
class DotSequence(ConditionMixin, dj.Computed):
    definition = ConditionMixin.definition.format(table="DotSequence")

    @staticmethod
    def _frames(condition_hash):
        key = dict(condition_hash=condition_hash)
        sequence = (stimulus.DotSequence & key).fetch1()
        return (dot.Trace * dot.Display & sequence).fetch1("n_frames")

    @property
    def frames(self):
        sequence = (stimulus.DotSequence * self).fetch1()

        images = (dot.Dot * dot.Sequence.Dot * dot.Display & sequence).fetch("image", order_by="dot_id")
        assert len(images) == sequence["sequence_length"]

        n_frames, id_trace = (dot.Trace * dot.Display & sequence).fetch1("n_frames", "id_trace")
        frames = np.stack(images[id_trace])
        assert len(frames) == n_frames

        return Video([Image.fromarray(frame, mode="L") for frame in frames])


# ---------- stimulus.RdkSequence ----------


@schema
class RdkSequence(ConditionMixin, dj.Computed):
    definition = ConditionMixin.definition.format(table="RdkSequence")

    @staticmethod
    def _frames(condition_hash):
        key = dict(condition_hash=condition_hash)
        sequence = (stimulus.RdkSequence & key).fetch1()

        keys = [dict(sequence, sequence_id=i) for i in range(sequence["sequence_length"])]
        tables = [
            rdk.RotationRdk * rdk.Sequence.Rotation,
            rdk.RadialRdk * rdk.Sequence.Radial,
            rdk.TranslationRdk * rdk.Sequence.Translation,
        ]
        frames = np.concatenate([(table & keys).fetch("n_frames") for table in tables])
        assert len(frames) == sequence["sequence_length"]

        return frames.sum()

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

        return Video([Image.fromarray(frame, mode="L") for frame in frames])


# ---------- stimulus.Frame ----------


@schema
class Frame(ConditionMixin, dj.Computed):
    definition = ConditionMixin.definition.format(table="Frame")

    @staticmethod
    def _frames(condition_hash):
        key = dict(condition_hash=condition_hash)
        pre_blank = (stimulus.Frame & key).fetch1("pre_blank_period") > 0

        if pre_blank:
            return 3
        else:
            return 2

    @property
    def frames(self):
        tup = stimulus.StaticImage.Image * stimulus.Frame & self
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
