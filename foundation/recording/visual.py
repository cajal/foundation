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
    -> utility.Offset
    -> utility.Impulse
    -> utility.Precision
    ---
    direction               : longblob         # presented directions (degrees, sorted) 
    response                : longblob         # response (STA) to directions
    density                 : longblob         # density of directions
    """

    @property
    def key_source(self):
        from foundation.recording.compute.visual import VisualDirectionTuning

        return VisualDirectionTuning.key_source

    def make(self, key):
        from foundation.recording.compute.visual import VisualDirectionTuning

        # visual direction tuning
        key["direction"], key["response"], key["density"] = (VisualDirectionTuning & key).tuning

        # insert
        self.insert1(key)


@schema.computed
class VisualSpatialTuning:
    definition = """
    -> Trace
    -> TrialFilterSet
    -> stimulus.VideoSet
    -> utility.Offset
    -> utility.Impulse
    -> utility.Resolution
    spatial_type            : varchar(128)      # spatial type
    ---
    response                : longblob         # response (STA) to spatial locations -- 2D array
    density                 : longblob         # density of spatial locations -- 2D array
    """

    @property
    def key_source(self):
        from foundation.recording.compute.visual import VisualSpatialTuning

        return VisualSpatialTuning.key_source

    def make(self, key):
        from foundation.recording.compute.visual import VisualSpatialTuning

        # rows for each spatial type
        rows = []

        # spatial tuning
        for spatial_type, response, density in (VisualSpatialTuning & key).tuning:

            # collect row
            rows.append(dict(key, spatial_type=spatial_type, response=response, density=density))

        # insert
        self.insert(rows)
