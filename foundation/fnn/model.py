from djutils import rowproperty
from foundation.fnn.network import Network, NetworkSet
from foundation.fnn.train import State, Loader, Objective, Optimizer, Scheduler
from foundation.schemas import fnn as schema


# -------------- Model --------------

# -- Model Base --


class _Model:
    """Model"""

    @rowproperty
    def networks(self):
        """
        Yields
        ------
        str
            network_id
        dict
            parameter state dict
        """
        raise NotImplementedError()


# -- Model Types --


@schema.lookup
class NetworkSetModel(_Model):
    definition = """
    -> NetworkSet
    -> State
    -> Loader
    -> Objective
    -> Optimizer
    -> Scheduler
    cycle           : int unsigned  # training cycle
    seed            : int unsigned  # seed for optimization
    instances       : int unsigned  # parallel training instances
    """

    @rowproperty
    def networks(self):
        from foundation.fnn.compute import TrainNetworkSet

        yield from (TrainNetworkSet & self).train()


@schema.lookup
class NetworkModel(_Model):
    definition = """
    -> Network
    -> State
    -> Loader
    -> Objective
    -> Optimizer
    -> Scheduler
    cycle           : int unsigned  # training cycle
    seed            : int unsigned  # seed for optimization
    instances       : int unsigned  # parallel training instances
    """

    @rowproperty
    def networks(self):
        from foundation.fnn.compute import TrainNetwork

        yield (TrainNetwork & self).train()


# -- Model --


@schema.link
class Model:
    links = [NetworkSetModel, NetworkModel]
    name = "model"
    comment = "neural network model"


# -- Computed Model --


@schema.computed
class ModelNetwork:
    definition = """
    -> Model
    -> Network
    ---
    parameters      : longblob      # parameter state dict
    """

    @property
    def key_source(self):
        return Model

    def make(self, key):
        from foundation.utils.torch import save_to_array

        for network_id, state_dict in (Model & key).link.networks:

            _key = dict(
                key,
                network_id=network_id,
                parameters=save_to_array(state_dict),
            )
            self.insert1(_key)
