from djutils import keys, rowmethod
from foundation.utils import logger
from foundation.virtual import fnn


# ----------------------------- Network State -----------------------------

# -- Network State Base --


@keys
class NetworkState:
    """Network State"""

    @property
    def key_list(self):
        return [fnn.Network] + self.state_list

    @property
    def state_list(self):
        raise NotImplementedError()

    @rowmethod
    def build(self, initialize=True):
        """
        Parameters
        ----------
        initialize : bool
            initialize network parameters

        Returns
        -------
        fnn.networks.Network
            network module
        """
        raise NotImplementedError()


# -- Network State Types --


class RandomNetwork(NetworkState):
    """Random Network State"""

    @property
    def state_list(self):
        return [fnn.RandomState]

    @rowmethod
    def build(self, initialize=True):
        import torch
        from foundation.fnn.network import Network

        devices = list(range(torch.cuda.device_count()))
        with torch.random.fork_rng(devices):

            if initialize:
                seed = self.key.fetch1("seed")
                torch.manual_seed(seed)
                logger.info(f"Initializing network with random seed {seed}")

            return (Network & self.key).link.module
