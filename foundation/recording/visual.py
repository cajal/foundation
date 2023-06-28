from djutils import rowproperty
from foundation.virtual import utility, stimulus
from foundation.recording.trial import Trial, TrialFilterSet
from foundation.recording.trace import Trace
from foundation.schemas import recording as schema


@schema.computed
class VisualMeasure:
    definition = """
    -> Trace
    -> TrialFilterSet
    -> stimulus.VideoSet
    -> utility.Resample
    -> utility.Offset
    -> utility.Rate
    -> utility.Measure
    -> utility.Burnin
    ---
    measure = NULL          : float     # visual response measure
    """

    def make(self, key):
        from foundation.recording.compute_visual import VisualMeasure

        key["measure"] = (VisualMeasure & key).measure
        self.insert1(key)
