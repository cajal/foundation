from djutils import rowproperty
from foundation.virtual import recording, fnn
from foundation.schemas import tuning as schema


# ---------------------------- Spatial Tuning ----------------------------

# -- Spatial Tuning Interface --


class SpatialType:
    """Spatial Type"""

    @rowproperty
    def compute(self):
        """
        Returns
        -------
        foundation.tuning.compute.spatial.SpatialType (row)
            compute spatial
        """
        raise NotImplementedError()


# -- Spatial Types --


@schema.lookup
class RecordingVisualSpatial(SpatialType):
    definition = """
    -> recording.VisualSpatialTuning
    """

    @rowproperty
    def compute(self):
        from foundation.tuning.compute.spatial import RecordingVisualSpatial

        return RecordingVisualSpatial & self


@schema.lookup
class FnnVisualSpatial(SpatialType):
    definition = """
    -> fnn.VisualSpatialTuning
    """

    @rowproperty
    def compute(self):
        from foundation.tuning.compute.spatial import FnnVisualSpatial

        return FnnVisualSpatial & self


# -- Spatial Tuning --


@schema.link
class Spatial:
    links = [RecordingVisualSpatial, FnnVisualSpatial]
    name = "spatial"
    comment = "spatial tuning"


# ----------------------------- Spatial Fit -----------------------------


@schema.computed
class SSI:
    definition = """
    -> Spatial
    ---
    ssi         : float     # spatial selectivity index
    """

    def make(self, key):
        from foundation.tuning.compute.spatial import SSI

        key["ssi"] = (SSI & key).ssi
        self.insert1(key)
