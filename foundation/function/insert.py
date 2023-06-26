from djutils import keys, merge, cache_rowproperty
from foundation.virtual import utility, stimulus, scan, recording, fnn, function


@keys
class VisualScanFnnCCNorm:
    """Visual Scan Fnn -- Normalized Signal Correlation"""

    @property
    def key_list(self):
        return [
            fnn.NetworkModel,
            fnn.VisualScanNetwork,
            recording.TrialFilterSet,
            utility.Bool.proj(trial_perspective="bool"),
            utility.Bool.proj(trial_modulation="bool"),
            stimulus.VideoSet,
            utility.Burnin,
        ]

    def fill(self, cuda=True):
        from contextlib import nullcontext
        from foundation.utils.torch import use_cuda
        from foundation.function.scan import VisualScanFnnRecordingResponse
        from foundation.function.response import Response, ResponseSet, VisualResponseMeasure, VisualResponseCorrelation

        # scan response
        VisualScanFnnRecordingResponse.populate(self.key, display_progress=True, reserve_jobs=True, limit=100)

        for key in self.key:

            # populate with caching and cuda
            with cache_rowproperty(), use_cuda() if cuda else nullcontext():

                # response key
                key = VisualScanFnnRecordingResponse & key

                # cc_abs
                VisualResponseCorrelation.populate(
                    self.key,
                    key,
                    utility.Correlation.CCSignal,
                    display_progress=True,
                    reserve_jobs=True,
                )

                # cc_max
                VisualResponseMeasure.populate(
                    self.key,
                    key * ResponseSet.Member & Response.RecordingResponse,
                    utility.Measure.CCMax,
                    display_progress=True,
                    reserve_jobs=True,
                )
