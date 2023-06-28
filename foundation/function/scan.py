from djutils import U
from foundation.virtual.bridge import pipe_fuse
from foundation.virtual import utility, recording, fnn
from foundation.function.response import TrialResponse, FnnTrialResponse, Response, ResponseSet
from foundation.schemas import function as schema


@schema.computed
class VisualScanFnnRecordingResponse:
    definition = """
    -> pipe_fuse.ScanSet.Unit                       # scan unit
    -> fnn.NetworkModel                             # network model
    -> recording.TrialFilterSet                     # trial filter
    -> utility.Bool.proj(trial_perspective="bool")  # trial | default perspective
    -> utility.Bool.proj(trial_modulation="bool")   # trial | default modulation
    ---
    -> ResponseSet                                  # fnn/recording response pair
    """

    @property
    def key_source(self):
        key = (
            fnn.NetworkModel
            * fnn.VisualScanNetwork
            * recording.ScanUnits
            * recording.TraceSet.Member
            * recording.Trace.ScanUnit
            * pipe_fuse.ScanSet.Unit
        ).proj()

        key *= recording.TrialFilterSet.proj()
        key *= utility.Bool.proj(trial_perspective="bool")
        key *= utility.Bool.proj(trial_modulation="bool")

        return U(*self.primary_key) & key

    def make(self, key):
        # unit key
        unit = (
            fnn.VisualScanNetwork
            * fnn.VisualScan
            * fnn.Spec.VisualSpec
            * fnn.VisualSpec
            * recording.ScanTrials
            * recording.ScanUnits
            * recording.TraceSet.Member
            * recording.Trace.ScanUnit
            * recording.ScanUnit
        )
        unit = unit.proj(
            "trialset_id",
            "traceset_index",
            data_filterset_id="trial_filterset_id",
            resample_id="unit_resample_id",
            offset_id="unit_offset_id",
            standardize_id="unit_standardize_id",
        )
        unit = (unit & key).fetch1()
        unit.update(response_index=unit["traceset_index"], **key)

        # insert responses
        TrialResponse.insert1(unit, ignore_extra_fields=True, skip_duplicates=True)
        FnnTrialResponse.insert1(unit, ignore_extra_fields=True, skip_duplicates=True)

        # fill responses
        Response.fill()

        # response pair
        pair = [Response.RecordingResponse, Response.FnnRecordingResponse]
        pair = [(table & unit).fetch1("KEY") for table in pair]
        pair = ResponseSet.fill(pair, prompt=False, silent=True)

        # insert key
        self.insert1(dict(key, **pair), ignore_extra_fields=True)
