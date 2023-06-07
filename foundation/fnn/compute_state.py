from djutils import keys, merge, rowmethod
from foundation.utils import logger
from foundation.virtual import fnn


# ----------------------------- Network State -----------------------------

# -- Network State Base --


class NetworkState:
    """Network State"""

    @rowmethod
    def build(self, network_id, initialize=True):
        """
        Parameters
        ----------
        network_id : str
            key -- foundation.fnn.network.Network
        initialize : bool
            initialize network parameters

        Returns
        -------
        fnn.networks.Network
            network module
        """
        raise NotImplementedError()


# -- Network State Types --


@keys
class RandomNetwork(NetworkState):
    """Random Network State"""

    @property
    def key_list(self):
        return [
            fnn.RandomState,
        ]

    @rowmethod
    def build(self, network_id, initialize=True):
        import torch
        from foundation.fnn.network import Network

        # fork torch rng
        devices = list(range(torch.cuda.device_count()))
        with torch.random.fork_rng(devices):

            # torch rng seed
            if initialize:
                seed = self.key.fetch1("seed")
                torch.manual_seed(seed)
                logger.info(f"Initializing network with random seed {seed}")

            # build module
            return (Network & {"network_id": network_id}).link.module


@keys
class NetworkSetCore(NetworkState):
    """Random Network State"""

    @property
    def key_list(self):
        return [
            fnn.NetworkSetCore,
        ]

    @rowmethod
    def build(self, network_id, initialize=True):
        from foundation.fnn.network import Network, NetworkSet
        from foundation.fnn.model import Model, ModelNetwork

        # network set
        networks = (NetworkSet & self.key).ordered_keys

        # make sure that network set core == network core
        (core_id,) = set(map(lambda k: (Network & k).link.fetch1("core_id"), networks))
        assert core_id == (Network & {"network_id": network_id}).link.fetch1("core_id")

        # build model
        module = RandomNetwork & self.key.proj(seed="state_seed")
        module = module.build(network_id=network_id, initialize=initialize)

        if initialize:
            # load model
            model = merge(self.key.proj(seed="model_seed"), Model.NetworkSetModel, ModelNetwork & networks[0])
            model = (ModelNetwork & model.proj()).model

            # transfer core
            module.core.load_state_dict(model.core.state_dict())
            logger.info(f"Transferring core parameters")

        if self.key.fetch1("freeze_core"):
            # freeze core
            module.core.freeze(True)
            logger.info("Freezing core")

        return module
