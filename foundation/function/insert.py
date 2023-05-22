from djutils import keys, merge, rowproperty, rowmethod
from foundation.virtual import utility, scan, recording, fnn


@keys
class VisualFnnScan:
    """Visual Fnn Scan Model"""

    @property
    def key_list(self):
        return [
            fnn.ModelNetwork * fnn.Network.VisualNetwork & fnn.VisualSet.VisualScan & fnn.VisualSpec.ResampleVisual,
        ]

    @property
    def responses(self):
        key = merge(
            self.key,
            fnn.Network.VisualNetwork,
            fnn.VisualSet.VisualScan,
            fnn.VisualSpec.ResampleVisual,
            fnn.VisualRecording,
        )
        key = key.proj(
            ...,
            resample_id="resample_id_u",
            offset_id="offset_id_u",
            traceset_id="traceset_id_u",
        )
        key = merge(
            key,
            utility.Resample,
            utility.Offset,
            utility.Rate,
            recording.TraceSet.Member,
        )
        return key.proj(response_index="traceset_index")

    def fill_responses(self):
        from foundation.function.response import Recording, Fnn, Response

        responses = self.responses
        Recording.insert(
            responses,
            ignore_extra_fields=True,
            skip_duplicates=True,
        )
        Fnn.insert(
            responses,
            ignore_extra_fields=True,
            skip_duplicates=True,
        )
        Response.fill()
