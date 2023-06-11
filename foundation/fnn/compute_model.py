from djutils import keys, merge, rowmethod
from foundation.utils import logger
from foundation.virtual import fnn


# ----------------------------- Network Model -----------------------------

# -- Network Model Base --


class NetworkModel:
    """Network Model"""

    @rowmethod
    def train(self, network_id):
        """
        Parameters
        ----------
        network_id : str
            key (foundation.fnn.network.Network)

        Yields
        ------
        str
            network_id
        dict
            network parameters (pytorch state dict)
        """
        raise NotImplementedError()


# -- Network Model Types --


@keys
class Instance(NetworkModel):
    """Network Model Instance"""

    @property
    def key_list(self):
        return [
            fnn.Instance,
        ]
