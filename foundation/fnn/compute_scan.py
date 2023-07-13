from djutils import keys, merge, rowproperty
from foundation.virtual import utility, stimulus, recording, fnn


@keys
class VisualScanPerformance:
    """Visual Scan CC Norm"""

    @property
    def keys(self):
        return [
            fnn.NetworkModel & (fnn.Network.VisualNetwork & fnn.Data.VisualScan),
            utility.Bool.proj(trial_perspective="bool"),
            utility.Bool.proj(trial_modulation="bool"),
            recording.TrialFilterSet,
            stimulus.VideoSet,
            utility.Burnin,
        ]

    def cc_norm(self):
        # key
        key = self.key.fetch(as_dict=True)

        # data
        data = fnn.Network.VisualNetwork.proj("data_id") * fnn.Data.VisualScan & key

        # data specfication
        spec = (fnn.Spec.VisualSpec & data).proj(
            "rate_id",
            resample_id="unit_resample_id",
            offset_id="unit_offset_id",
            standardize_id="unit_standardize_id",
        )

        # traces
        traces = data * recording.ScanUnits * recording.TraceSet.Member * recording.Trace.ScanUnit * recording.ScanUnit
        traces = traces.proj("traceset_index")

        # CC max
        cc_max = (recording.VisualMeasure & utility.Measure.CCMax & key & spec) * traces
        cc_max = cc_max.proj("traceset_index", cc_max="measure")

        # CC abs
        cc_abs = fnn.VisualUnitCorrelation & utility.Correlation.CCSignal & key
        cc_abs = cc_abs.proj(cc_abs="correlation", traceset_index="unit_index")

        # CC norm
        return (cc_abs * cc_max).proj("cc_abs", "cc_max", cc_norm="cc_abs / cc_max")
