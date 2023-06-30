from djutils import rowproperty
from foundation.virtual import utility, stimulus, recording
from foundation.fnn.network import NetworkUnit
from foundation.fnn.model import NetworkModel
from foundation.schemas import fnn as schema


@schema.computed
class VisualUnitCorrelation:
    definition = """
    -> NetworkModel
    -> NetworkUnit
    -> utility.Bool.proj(trial_perspective="bool")
    -> utility.Bool.proj(trial_modulation="bool")
    -> recording.TrialFilterSet
    -> stimulus.VideoSet
    -> utility.Correlation
    -> utility.Burnin
    ---
    correlation = NULL      : float     # visual response correlation
    """

    @property
    def key_source(self):
        from foundation.fnn.compute_visual import VisualUnitCorrelation

        return VisualUnitCorrelation.key_source

    def make(self, key):
        from foundation.fnn.compute_visual import VisualUnitCorrelation

        key["correlation"] = (VisualUnitCorrelation & key).correlation
        self.insert1(key)
