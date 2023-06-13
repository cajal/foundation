from djutils import keys, merge, cache_rowproperty
from foundation.virtual import utility, stimulus, scan, recording, fnn


@keys
class FnnVisualScanCCNorm:
    """Fnn Visual Scan -- Normalized Signal Correlation"""

    @property
    def key_list(self):
        return [
            fnn.NetworkModel,
            fnn.VisualScaNetwork,
            recording.TrialFilterSet,
            utility.Bool.proj(perspectives="bool"),
            utility.Bool.proj(modulations="bool"),
            stimulus.VideoSet,
            utility.Burnin,
        ]

    def fill(self, cuda=True):
        from contextlib import nullcontext
        from foundation.utils.torch import use_cuda
        from foundation.function.scan import FnnVisualScanTrialResponse
        from foundation.function.response import Response, ResponseSet, VisualResponseMeasure, VisualResponseCorrelation

        # scan response
        FnnVisualScanTrialResponse.populate(self.key, display_progress=True, reserve_jobs=True)

        # scan response keys
        keys = fnn.VisualScanModel.proj() * recording.TrialFilterSet.proj() & self.key
        for key in keys:
            with cache_rowproperty(), use_cuda() if cuda else nullcontext():
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
                    display_progress=True,
                    reserve_jobs=True,
                )
