from djutils import keys, rowmethod
from foundation.utils import logger
from foundation.virtual import fnn


# ----------------------------- Transfer -----------------------------

# -- Transfer Interface --


class TransferType:
    """Transfer"""

    @rowmethod
    def transfer(self, transferlist_id, network_id, network):
        """
        Parameters
        ----------
        transferlist_id : str
            key (foundation.fnn.transfer.TransferList)
        network_id : str
            key (foundation.fnn.network.Network)
        network : fnn.networks.Network
            network module

        Returns
        -------
        fnn.networks.Network
            network module with transferred parameters
        """
        raise NotImplementedError()


# -- Transfer Types --


@keys
class FoundationTransfer(TransferType):
    """Foundation Transfer"""

    @property
    def keys(self):
        return [
            fnn.FoundationTransfer,
        ]

    @rowmethod
    def transfer(self, transferlist_id, network_id, network):
        from foundation.fnn.data import DataSet
        from foundation.fnn.network import ModuleSet
        from foundation.fnn.instance import Instance
        from foundation.fnn.model import Model

        # data key
        dkey = (DataSet & self.item).members.fetch("KEY", order_by="dataset_index", limit=1)

        # network key
        nkey = {"network_id": network_id}

        # instance key
        tkey = {"transferlist_id": transferlist_id}
        ikey = (Instance.Foundation & self.item & tkey).fetch1("KEY")

        # load model
        model = (Model & dkey & nkey & ikey).model()

        # transfer modules
        modules = (ModuleSet & self.item).members.fetch("module", order_by="moduleset_index")

        for module in modules:

            _model = getattr(model, module)
            _network = getattr(network, module)

            # transfer parameters
            _network.load_state_dict(_model.state_dict())

            # freeze parameters
            _network.freeze(self.item["freeze"])

        return network
