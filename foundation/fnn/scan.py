from djutils import merge
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
        # scan key
        net = Network.VisualNetwork & key
        data = Data.VisualScan & net
        scan = (VisualScan & data).fetch1()

        # insert
        self.insert1(dict(key, **scan))
