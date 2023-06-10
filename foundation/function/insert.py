from djutils import keys, merge, cache_rowproperty
from foundation.virtual import utility, stimulus, scan, recording, fnn


@keys
class FnnVisualScanCCNorm:
    """Fnn Visual Scan -- Normalized Signal Correlation"""

    @property
    def key_list(self):
        return [
            fnn.VisualScanModel,
            recording.TrialFilterSet,
            stimulus.VideoSet,
            utility.Bool.proj(perspective="bool"),
            utility.Bool.proj(modulation="bool"),
            utility.Burnin,
        ]

    def fill(self):
        from foundation.utils.torch import use_cuda
        from foundation.function.scan import VisualScanFnnTrialResponse
        from foundation.function.response import Response, ResponseSet, VisualResponseMeasure, VisualResponseCorrelation

        # scan response
        VisualScanFnnTrialResponse.populate(self.key, display_progress=True, reserve_jobs=True)

        # scan response keys
        keys = fnn.VisualScanModel.proj() * recording.TrialFilterSet.proj() & self.key
        for key in keys:
            with cache_rowproperty(), use_cuda():
                # populate with caching and cuda
                key = merge(
                    fnn.VisualScanModel.proj() & key,
                    fnn.VisualScanResponse,
                    VisualScanFnnTrialResponse & self.key,
                )
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
                    key * ResponseSet.Member & Response.TrialResponse,
                    utility.Measure.CCMax,
                )
