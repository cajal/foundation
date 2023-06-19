from djutils import merge
from foundation.virtual.bridge import pipe_fuse
from foundation.fnn.network import Network
from foundation.fnn.data import Data, VisualScan
from foundation.schemas import fnn as schema


@schema.computed
class VisualScanNetwork:
    definition = """
    -> Network
    ---
    -> VisualScan
    """

    @property
    def key_source(self):
        return (Network.VisualNetwork & Data.VisualScan).proj()

    def make(self, key):
        # scan data
        data = (Network & key).link.data.key.fetch1()

        # insert
        self.insert1(dict(key, **data))


@schema.computed
class VisualScanUnit:
    definition = """
    -> VisualScanNetwork
    response_index      : int unsigned  # network response index
    ---
    -> pipe_fuse.ScanSet.Unit
    """

    def make(self, key):
        from foundation.recording.scan import ScanUnits
        from foundation.recording.trace import Trace, TraceSet

        # scan units
        units = (Network & key).link.data
        units = ScanUnits & units.key
        units = (TraceSet & units).members
        units = merge(units, Trace.ScanUnit)

        # unit keys
        keys = (VisualScanNetwork & key).proj() * units.proj(..., response_index="traceset_index")

        # insert
        self.insert(keys, ignore_extra_fields=True)
