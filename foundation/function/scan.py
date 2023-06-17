from djutils import merge
from foundation.virtual import recording, fnn
from foundation.function.response import (
    TrialResponse,
    TrialPerspectives,
    TrialModulations,
    FnnTrialResponse,
    Response,
    ResponseSet,
)
from foundation.schemas import function as schema


@schema.computed
class FnnVisualScanTrialResponse:
    definition = """
    -> fnn.NetworkModel             # network model
    -> recording.TrialFilterSet     # trial filter
    -> TrialPerspectives            # trial perspectives
    -> TrialModulations             # trial modulations
    -> fnn.VisualScanUnit           # scan unit
    ---
    -> ResponseSet                  # model/recording response pair
    """

    def make(self, key):
        # unit key
        unit_key = merge(
            fnn.VisualScanUnit & key,
            recording.ScanUnit,
            recording.Trace.ScanUnit,
            recording.ScanTrials,
            recording.TrialSet,
            fnn.VisualScanNetwork,
            fnn.VisualScan,
            fnn.Spec.VisualSpec,
            fnn.VisualSpec,
        )
        unit_key = unit_key.proj(
            resample_id="unit_resample_id",
            offset_id="unit_offset_id",
            standardize_id="unit_standardize_id",
        )
        unit_key = unit_key.fetch1()
        unit_key.update(key)

        # insert responses
        TrialResponse.insert1(unit_key, ignore_extra_fields=True, skip_duplicates=True)
        FnnTrialResponse.insert1(unit_key, ignore_extra_fields=True, skip_duplicates=True)

        # fill responses
        Response.fill()

        # response pair
        pair = [Response.TrialResponse, Response.FnnTrialResponse]
        pair = [(table & unit_key).fetch1("KEY") for table in pair]
        pair = ResponseSet.fill(pair, prompt=False, silent=True)

        # insert key
        self.insert1(dict(key, **pair), ignore_extra_fields=True)
