from djutils import keys, merge, rowmethod
from foundation.utils import logger
from foundation.virtual import fnn


# ----------------------------- State -----------------------------

# -- State Interface --


class StateType:
    """Network State"""

    @rowmethod
    def build(self, network_id, initialize=True):
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
    """Initial Network State"""

    @property
    def keys(self):
        return [
            fnn.Initial,
        ]

    @rowmethod
    def build(self, network_id, initialize=True):
        from foundation.utils import torch_rng
        from foundation.fnn.network import Network

        if initialize:
            # set manual seed for parameter initialization
            seed = self.key.fetch1("seed")
            logger.info(f"Initializing network with random seed {seed}")
        else:
            # no need to set manual seed
            seed = None

        # fork torch rng
        with torch_rng(seed=seed):

            # build module
            return (Network & {"network_id": network_id}).link.module
