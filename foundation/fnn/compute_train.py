from djutils import keys, rowmethod
from foundation.utils import logger
from foundation.virtual import fnn


# ----------------------------- Train -----------------------------

# -- Train Interface --


class TrainType:
    """Train Network"""

    @rowmethod
    def train(self, network_id, data_id, rank=0, cycle=0, checkpoint=None):
        """
        Parameters
        ----------
        fnn.networks.Network
            network module
        rank : int
            distributed rank
        cycle : int
            training cycle
        checkpont : dict | None
            training checkpoint

        Yields
        ------
        epoch : int
            epoch number
        info : dict
            training info
        checkpoint : dict
            training checkpoint
        """
        raise NotImplementedError()


# -- Train Types --


@keys
class Parallel(TrainType):
    @property
    def keys(self):
        return [
            fnn.Parallel,
        ]

    @rowmethod
    def train(self, network, rank=0, cycle=0, checkpoint=None):
        from foundation.fnn.train import State, Scheduler, Optimizer, Loader, Objective
        from foundation.fnn.data import Data

        # # network state
        # network = (State & self.item).link.compute.state(network_id=network_id, data_id=data_id)
        # network = network.to(device="cuda")
