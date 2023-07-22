from djutils import keys, merge, rowmethod
from foundation.utils import logger
from foundation.virtual import fnn


# ----------------------------- State -----------------------------

# -- State Interface --


class StateType:
    """Network State"""

    @rowmethod
    def state(self, network_id, initialize=True):
        """
        Parameters
        ----------
        network_id : str
            key (foundation.fnn.network.Network)
        initialize : bool
            initialize network parameters

        Returns
        -------
        fnn.networks.Network
            network module
        """
        raise NotImplementedError()


# -- State Types --


@keys
class Initial(StateType):
    """Initial Network"""

    @property
    def keys(self):
        return [
            fnn.Initial,
        ]

    @rowmethod
    def state(self, network_id):
        from foundation.utils import torch_rng
        from foundation.fnn.network import Network

        # fork torch rng
        with torch_rng(seed=self.item["seed"]):
            logger.info(f"Initializing parameters with random seed {self.item['seed']}")

            # build module
            return (Network & {"network_id": network_id}).link.module


@keys
class SharedCore(StateType):
    """Shared Core"""

    @property
    def keys(self):
        models = fnn.NetworkSetModel & {"module": "core", "shared": True}
        return [
            fnn.SharedCore & models & fnn.NetworkSetCore,
        ]

    @rowmethod
    def state(self, network_id):
        from foundation.utils import torch_rng
        from foundation.fnn.model import NetworkModel
        from foundation.fnn.network import Network, NetworkSet, NetworkSetCore

        # network
        net = (Network & {"network_id": network_id}).link

        # ensure core_id and streams match
        assert net.fetch1("core_id", "streams") == (NetworkSetCore & self.item).fetch1("core_id", "streams")

        with torch_rng(seed=self.item["seed"]):
            logger.info(f"Initializing non-core parameters with random seed {self.item['seed']}")

            # build module
            network = net.module

        logger.info(f"Transferring core parameters from {self.item['networkset_id']}")

        # load model
        nets = (NetworkSet & self.item).members
        key = nets & {"networkset_index": 0}
        model = (NetworkModel & key & self.item).model

        # transfer parameters
        network.core.load_state_dict(model.core.state_dict())

        if self.item["freeze"]:
            logger.info("Freezing core")

            # freeze core
            network.core.freeze(True)

        return network
