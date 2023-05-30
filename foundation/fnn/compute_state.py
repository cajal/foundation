from djutils import keys, rowmethod
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
        return [fnn.RandomState]

    @rowmethod
    def build(self, network_id, initialize=True):
        import torch
        from foundation.fnn.network import Network

        devices = list(range(torch.cuda.device_count()))
        with torch.random.fork_rng(devices):

            if initialize:
                seed = self.key.fetch1("seed")
                torch.manual_seed(seed)
                logger.info(f"Initializing network with random seed {seed}")

            return (Network & {"network_id": network_id}).link.module
