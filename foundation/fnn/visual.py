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
        correlations = (VisualRecordingCorrelation & key).correlation()

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

        # visual direction tuning
        key["direction"], response, density = (VisualDirectionTuning & key).tuning()

        # verify units
        assert len(response) == len(density) == (Data & key).link.compute.units

        # create keys
        keys = [dict(key, unit=u, response=r, density=d) for u, (r, d) in enumerate(zip(response, density))]

        # insert
        self.insert(keys)


@schema.computed
class VisualSpatialTuning:
    definition = """
    -> Model
    -> stimulus.VideoSet
    -> utility.Offset
    -> utility.Impulse
    -> utility.Resolution
    -> utility.Burnin
    unit                    : int unsigned     # unit index
    spatial_type            : varchar(128)     # spatial type    
    ---
    response                : longblob         # response (STA) to spatial locations -- 2D array
    density                 : longblob         # density of spatial locations -- 2D array
    """

    @property
    def key_source(self):
        from foundation.fnn.compute.visual import VisualSpatialTuning

        return VisualSpatialTuning.key_source

    def make(self, key):
        from foundation.fnn.compute.visual import VisualSpatialTuning

        # number of units
        units = (Data & key).link.compute.units

        # spatial tuning
        for spatial_type, response, density in (VisualSpatialTuning & key).tuning():

            # verify units
            assert len(response) == len(density) == units

            # create keys
            keys = [
                dict(key, spatial_type=spatial_type, unit=u, response=r, density=d)
                for u, (r, d) in enumerate(zip(response, density))
            ]

            # insert
            self.insert(keys)
