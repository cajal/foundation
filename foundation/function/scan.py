from djutils import merge
from foundation.virtual import utility, recording, fnn
from foundation.function.response import RecordingResponse, FnnRecordingResponse, Response, ResponseSet
from foundation.schemas import function as schema


@schema.computed
class VisualScanFnnRecordingResponse:
    definition = """
    -> fnn.NetworkModel
    -> recording.TrialFilterSet
    -> utility.Bool.proj(trial_perspectives="bool")
    -> utility.Bool.proj(trial_modulations="bool")
    -> fnn.VisualScanUnit
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
        RecordingResponse.insert1(unit_key, ignore_extra_fields=True, skip_duplicates=True)
        FnnRecordingResponse.insert1(unit_key, ignore_extra_fields=True, skip_duplicates=True)

        # fill responses
        Response.fill()

        # response pair
        pair = [Response.RecordingResponse, Response.FnnRecordingResponse]
        pair = [(table & unit_key).fetch1("KEY") for table in pair]
        pair = ResponseSet.fill(pair, prompt=False, silent=True)

        # insert key
        self.insert1(dict(key, **pair), ignore_extra_fields=True)
