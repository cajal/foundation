from djutils import keys, merge, cache_rowproperty
from foundation.virtual import utility, stimulus, scan, recording, fnn, function


@keys
class FnnVisualScanCCNorm:
    """Fnn Visual Scan -- Normalized Signal Correlation"""

    @property
    def key_list(self):
        return [
            fnn.NetworkModel,
            fnn.VisualScanNetwork,
            recording.TrialFilterSet,
            utility.Bool.proj(trial_perspectives="bool"),
            utility.Bool.proj(trial_modulations="bool"),
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

        for key in self.key:
            # populate with caching and cuda
            with cache_rowproperty(), use_cuda() if cuda else nullcontext():

                # response key
                key = merge(
                    fnn.VisualScanNetwork.proj() & key,
                    fnn.VisualScanUnit,
                    FnnVisualScanTrialResponse & self.key,
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
