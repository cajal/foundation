from djutils import rowproperty
from foundation.virtual import utility, stimulus, recording
from foundation.fnn.data import Data
from foundation.fnn.model import Model
from foundation.schemas import fnn as schema


@schema.computed
class VisualRecordingCorrelation:
    definition = """
    -> Model
    -> recording.TrialFilterSet
    -> stimulus.VideoSet
    -> utility.Correlation
    -> utility.Burnin
    -> utility.Bool.proj(perspective="bool")
    -> utility.Bool.proj(modulation="bool")
    unit                    : int unsigned      # unit index
    ---
    correlation = NULL      : float             # unit correlation
    """

    @property
    def key_source(self):
        from foundation.fnn.compute.visual import VisualRecordingCorrelation

        return VisualRecordingCorrelation.key_source

    def make(self, key):
        from foundation.fnn.compute.visual import VisualRecordingCorrelation

        # unit correlations
        correlations = (VisualRecordingCorrelation & key).units

        # verify units
        assert len(correlations) == (Data & key).link.compute.units

        # insert
        keys = [dict(key, unit_index=i, correlation=c) for i, c in enumerate(correlations)]
        self.insert1(keys)
