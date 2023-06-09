from djutils import merge
from foundation.virtual import recording, fnn
from foundation.function.response import TrialResponse, FnnTrialResponse, Response, ResponseSet
from foundation.schemas import function as schema


@schema.computed
class VisualScanFnnTrialResponse:
    definition = """
    -> fnn.VisualScanResponse
    -> recording.TrialFilterSet
    perspective                 : bool          # use recording perspective
    modulation                  : bool          # use recording modulation
    ---
    -> ResponseSet
    """

    def make(self, key):
        # unit key
        unit_key = merge(
            fnn.VisualScanResponse & key,
            recording.ScanUnit,
            recording.Trace.ScanUnit,
            recording.ScanTrials,
        )
        unit_key = (unit_key * fnn.VisualScanModel).proj(..., spec_id="units_id")
        unit_key = (unit_key * fnn.Spec.TraceSpec).fetch1()
        unit_key.update(key)

        # insert
        TrialResponse.insert1(unit_key, ignore_extra_fields=True, skip_duplicates=True)

        # fnn keys
        keys = []
        for perspective in [True, False]:
            for modulation in [True, False]:
                key = dict(unit_key, perspective=perspective, modulation=modulation)
                keys.append(key)

                # insert
                FnnTrialResponse.insert1(key, ignore_extra_fields=True, skip_duplicates=True)

        # fill response
        Response.fill()

        # response pairs
        for key in keys:
            pair = [
                (Response.TrialResponse & unit_key).fetch1("KEY"),
                (Response.FnnTrialResponse & key).fetch1("KEY"),
            ]
            pair = ResponseSet.fill(pair, prompt=False, silent=True)

            # insert
            self.insert1(dict(key, **pair), ignore_extra_fields=True)
