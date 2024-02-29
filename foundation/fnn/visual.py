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
        keys = [dict(key, unit=u, correlation=c) for u, c in enumerate(correlations)]
        self.insert(keys)


@schema.computed
class VisualDirectionTuning:
    definition = """
    -> Model
    -> stimulus.VideoSet
    -> utility.Offset
    -> utility.Impulse
    -> utility.Precision
    -> utility.Burnin
    unit                     : int unsigned     # unit index    
    ---
    direction                : longblob         # presented directions (degrees, sorted) 
    response                 : longblob         # response (STA) to directions
    density                  : longblob         # density of directions
    """

    @property
    def key_source(self):
        from foundation.fnn.compute.visual import VisualDirectionTuning

        return VisualDirectionTuning.key_source

    def make(self, key):
        from foundation.fnn.compute.visual import VisualDirectionTuning
        from itertools import repeat

        # unit tunings
        direction, mean, n_trials = (VisualDirectionTuning & key).tunings

        # verify units
        assert mean.shape[1] == (Data & key).link.compute.units

        # insert
        self.insert(
            {**key, "direction": d, "mean": m, "n_trials": n, "unit": u}
            for u, (d, m, n) in enumerate(zip(repeat(direction), mean.T, repeat(n_trials)))
        )
