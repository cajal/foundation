from djutils import keys, merge
from foundation.virtual import utility, stimulus, scan, recording, fnn


@keys
class VisualScanCorrelation:
    """Visual Scan Correlation"""

    @property
    def keys(self):
        return [
            (scan.Scan * fnn.Model) & fnn.Data.VisualScan,
            recording.TrialFilterSet,
            stimulus.VideoSet,
            utility.Burnin,
            utility.Bool.proj(perspective="bool"),
            utility.Bool.proj(modulation="bool"),
        ]

    def cc_norm(self):
        spec = fnn.Data.VisualScan.proj(
            "spec_id", "trace_filterset_id", "pipe_version", "segmentation_method", "spike_method"
        )
        units = merge(
            self.key,
            spec,
            fnn.Spec.VisualSpec.proj("rate_id", offset_id="offset_id_unit", resample_id="resample_id_unit"),
            (fnn.VisualRecordingCorrelation & utility.Correlation.CCSignal).proj(..., trace_order="unit"),
            recording.ScanUnitOrder,
            recording.Trace.ScanUnit,
            recording.VisualMeasure & utility.Measure.CCMax,
        )
        return units.proj(
            cc_abs="correlation",
            cc_max="measure",
            cc_norm="correlation/measure",
        )
