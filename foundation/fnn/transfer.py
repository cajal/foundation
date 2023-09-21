from djutils import rowproperty
from foundation.fnn.network import ModuleSet
from foundation.fnn.data import DataSet
from foundation.fnn.train import Train
from foundation.schemas import fnn as schema


# ----------------------------- Transfer -----------------------------

# -- Transfer Interface --


class TransferType:
    """Transfer"""

    @rowproperty
    def compute(self):
        """
        Returns
        -------
        foundation.fnn.compute.transfer.TransferType (row)
            compute transfer
        """
        raise NotImplementedError()


# -- Transfer Types --


@schema.lookup
class FoundationTransfer:
    definition = """
    -> ModuleSet
    -> DataSet
    -> Train
    parallel        : int unsigned      # parallel group size
    cycle           : int unsigned      # training cycle
    seed            : int unsigned      # seed for initialization
    """

    @rowproperty
    def compute(self):
        from foundation.fnn.compute.transfer import FoundationTransfer

        return FoundationTransfer & self


# -- Transfer --


@schema.link
class Transfer:
    links = [FoundationTransfer]
    name = "transfer"
    comment = "fnn transfer"


@schema.linklist
class TransferList:
    link = Transfer
    name = "transferlist"
    comment = "fnn transfer list"
