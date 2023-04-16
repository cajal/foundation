import numpy as np
import datajoint as dj
from djutils import link
from .conditions import Clip, Monet2, Trippy, GaborSequence, DotSequence, RdkSequence, Frame
from .transform import Resize


schema = dj.schema("foundation_stimuli")


# ---------- Frames Links ----------


@link(schema)
class Frames:
    links = [Clip, Monet2, Trippy, GaborSequence, DotSequence, RdkSequence, Frame]
    name = "frames"
    comment = "stimulus frames"
    length = 16


# ---------- Transform Links ----------


@link(schema)
class Transform:
    links = [Resize]
    name = "transform"
    comment = "stimulus transformation"
    length = 16


# ---------- Transformed Frames ----------


@schema
class TransformedFrames(dj.Computed):
    definition = """
    -> Frames
    -> Transform
    ---
    frames  : longblob          # stimulus frames [t, h, w, c]
    t       : int unsigned      # number of frames
    h       : int unsigned      # frame height
    w       : int unsigned      # frame width
    c       : int unsigned      # number of channels
    """

    def make(self, key):
        frames = (Frames & key).link.frames
        transform = (Transform & key).link.transform

        key["frames"] = transform(frames).array
        key["t"], key["h"], key["w"], key["c"] = key["frames"].shape

        self.insert1(key)
