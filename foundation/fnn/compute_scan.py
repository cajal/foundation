from djutils import keys, merge, rowproperty
from foundation.virtual import recording, fnn


@keys
class VisualScanNetwork:
    """Visual Scan Network"""

    @property
    def keys(self):
        return [
            fnn.VisualScanNetwork,
        ]

    @rowproperty
    def unit_traces(self):
        from foundation.recording.trace import Trace, TraceSet

        # unit traceset key
        key = merge(self.key, fnn.VisualScanNetwork, recording.ScanUnits)

        # traceset members
        units = (TraceSet & key).members

        # unit traces
        return merge(units, Trace.ScanUnit)
