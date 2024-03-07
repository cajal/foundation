import pandas as pd
from djutils import keys, rowproperty
from foundation.virtual import utility, stimulus, scan, recording, fnn


@keys
class VisualScanRecording:
    """Visual Scan Recording"""

    @property
    def keys(self):
        return [
            (scan.Scan * fnn.Data) & fnn.Data.VisualScan,
        ]

    @property
    def units(self):
        key = self.key.fetch("KEY")
        units = (
            fnn.Data.VisualScan * recording.ScanUnitOrder * recording.Trace.ScanUnit * recording.ScanUnit
            & key
        )
        return units.proj("trace_order")


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

    @rowproperty
    def cc_norm(self):
        from foundation.fnn.data import Data

        data = (Data & self.item).link.compute

        ukey = data.key_unit

        mkey = {
            "rate_id": ukey["rate_id"],
            "resample_id": ukey["resample_id"],
            "offset_id": ukey["offset_id"],
            "measure_id": utility.Measure.CCMax.fetch1("measure_id"),
        }

        ckey = {
            "correlation_id": utility.Correlation.CCSignal.fetch1("correlation_id"),
        }

        traces = recording.ScanUnitOrder & ukey

        measures = (recording.VisualMeasure * traces & mkey & self.item).proj(
            cc_max="measure", unit="trace_order"
        )

        correlations = (fnn.VisualRecordingCorrelation & ckey & self.item).proj(cc_abs="correlation")
        units = correlations * measures * recording.Trace.ScanUnit
        assert len(units) == data.units

        cols = [
            "animal_id",
            "session",
            "scan_idx",
            "pipe_version",
            "segmentation_method",
            "spike_method",
            "unit_id",
            "cc_abs",
            "cc_max",
        ]
        df = pd.DataFrame(units.fetch(*cols, as_dict=True))[cols]
        df["cc_norm"] = df.cc_abs / df.cc_max

        return df
