from djutils import rowproperty, Filepath
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
        fnn.modules.Module
            network module
        """
        raise NotImplementedError()


# -- Model Types --


@schema.lookup
class VisualModel(_Model):
    definition = """
    -> NetworkSet
    -> State
    -> Scheduler
    -> Loader
    -> Objective
    -> Optimizer
    seed            : int unsigned  # seed for optimization
    cycle           : int unsigned  # training cycle
    instances       : int unsigned  # parallel training instances
    """

    @rowproperty
    def networks(self):
        from foundation.fnn.compute import TrainVisualModel

        yield from (TrainVisualModel & self).train()


# -- Model --


@schema.link
class Model:
    links = [VisualModel]
    name = "model"
    comment = "neural network model"


# -- Computed Model --


@schema.computed
class NetworkModel(Filepath):
    definition = """
    -> Network
    -> Model
    ---
    parameters      : filepath@scratch09    # parameter state dict
    """

    @property
    def key_source(self):
        return Model

    def make(self, key):
        from torch import save

        for network_id, state_dict in (Model & key).link.networks:

            _key = dict(
                key,
                network_id=network_id,
            )
            filepath = self.createpath(
                _key,
                "parameters",
                suffix="pt",
            )
            save(
                state_dict,
                filepath,
                pickle_protocol=5,
            )
            self.insert1(dict(_key, parameters=filepath))
