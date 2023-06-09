from djutils import merge, rowproperty
from foundation.virtual.bridge import pipe_fuse
from foundation.fnn.model import ModelNetwork
from foundation.fnn.network import Network
from foundation.fnn.data import Data, VisualScan
from foundation.schemas import fnn as schema


@schema.computed
class VisualScanModel:
    definition = """
    -> ModelNetwork
    ---
    -> VisualScan
    """

    @property
    def key_source(self):
        keys = ModelNetwork & (Network.VisualNetwork & Data.VisualScan)
        return keys.proj()

    def make(self, key):
        # scan data
        data = (Network & key).link.data
        data = data.link.fetch1()

        # insert
        self.insert1(dict(key, **data))


@schema.computed
class VisualScanResponse:
    definition = """
    -> VisualScanModel
    response_index              : int unsigned      # network response index
    ---
    -> pipe_fuse.ScanSet.Unit
    """

    def make(self, key):
        from foundation.recording.scan import ScanUnits
        from foundation.recording.trace import Trace, TraceSet

        # scan units
        units = (Network & key).link.data
        units = ScanUnits & units.link
        units = (TraceSet & units).members
        units = merge(units, Trace.ScanUnit)

        # unit keys
        keys = (VisualScanModel & key).proj() * units.proj(..., response_index="traceset_index")

        # insert
        self.insert(keys, ignore_extra_fields=True)
