from djutils import merge, rowproperty
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


# @schema.computed
# class VisualScanModel:
#     definition = """
#     -> ModelNetwork
#     ---
#     -> VisualScan
#     """