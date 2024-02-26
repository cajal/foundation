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
    measure = NULL      : float     # visual response measure
    """

    @property
    def key_source(self):
        from foundation.recording.compute.visual import VisualMeasure

        return VisualMeasure.key_source

    def make(self, key):
        from foundation.recording.compute.visual import VisualMeasure

        # visual measure
        key["measure"] = (VisualMeasure & key).measure

        # insert
        self.insert1(key)


@schema.computed
class VisualDirectionTuning:
    definition = """
    -> Trace
    -> TrialFilterSet
    -> stimulus.VideoSet
    -> utility.Impulse
    -> utility.Precision
    -> utility.Offset
    ---
    direction                : longblob         # presented directions (degrees, sorted) 
    response                 : longblob         # repsonse to presented directions
    density                  : longblob         # density of presented directions
    """

    @property
    def key_source(self):
        from foundation.recording.compute.visual import VisualDirectionTuning

        return VisualDirectionTuning.key_source

    def make(self, key):
        from foundation.recording.compute.visual import VisualDirectionTuning

        # visual direction tuning
        key["directions"], key["mean"], key["n_trials"] = (VisualDirectionTuning & key).tuning

        # insert
        self.insert1(key)
