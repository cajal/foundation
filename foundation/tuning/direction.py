from djutils import rowproperty
from foundation.virtual import recording, fnn
from foundation.schemas import tuning as schema


# ---------------------------- Direction Tuning ----------------------------

# -- Direction Tuning Interface --


class DirectionType:
    """Direction Type"""

    @rowproperty
    def compute(self):
        """
        Returns
        -------
        foundation.tuning.compute.direction.DirectionType (row)
            compute direction
        """
        raise NotImplementedError()


# -- Direction Types --


@schema.lookup
class RecordingVisualDirection(DirectionType):
    definition = """
    -> recording.VisualDirectionTuning
    """

    @rowproperty
    def compute(self):
        from foundation.tuning.compute.direction import RecordingVisualDirection

        return RecordingVisualDirection & self


@schema.lookup
class FnnVisualDirection(DirectionType):
    definition = """
    -> fnn.VisualDirectionTuning
    """

    @rowproperty
    def compute(self):
        from foundation.tuning.compute.direction import FnnVisualDirection

        return FnnVisualDirection & self


# -- Direction Tuning --


@schema.link
class Direction:
    links = [RecordingVisualDirection, FnnVisualDirection]
    name = "direction"
    comment = "direction tuning"


# ----------------------------- Direction Fit -----------------------------


@schema.computed
class GlobalOSI:
    definition = """
    -> Direction
    ---
    global_osi      : float     # global orientation selectivity index
    """

    def make(self, key):
        from foundation.tuning.compute.direction import GlobalOSI

        key["global_osi"] = (GlobalOSI & key).global_osi
        self.insert1(key)


@schema.computed
class GlobalDSI:
    definition = """
    -> Direction
    ---
    global_dsi      : float     # global direction selectivity index
    """

    def make(self, key):
        from foundation.tuning.compute.direction import GlobalDSI

        key["global_dsi"] = (GlobalDSI & key).global_osi
        self.insert1(key)


@schema.computed
class BiVonMises:
    definition = """
    -> Direction
    ---
    success     : bool      # success of least squares optimization
    mu          : float     # center of the first von Mises distribution
    phi         : float     # weight of the first von Mises distribution
    kappa       : float     # dispersion of both von Mises distributions
    scale       : float     # von Mises amplitude
    bias        : float     # uniform amplitude
    mse         : float     # mean squared error
    """

    def make(self, key):
        from foundation.tuning.compute.direction import BiVonMises

        key = dict(key, **(BiVonMises & key).bi_von_mises)
        self.insert1(key)
