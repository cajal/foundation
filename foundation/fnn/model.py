from djutils import rowproperty, rowmethod
from foundation.fnn.network import Network, NetworkSet
from foundation.fnn.train import State, Loader, Objective, Optimizer, Scheduler
from foundation.schemas import fnn as schema


# ----------------------------- Model -----------------------------

# -- Model Interface --


class ModelType:
    """Model"""

    @rowproperty
    def compute(self):
        """
        Returns
        -------
        foundation.fnn.compute_model.NetworkModel (row)
            compute model
        """
        raise NotImplementedError()


# -- Model Types --


@schema.lookup
class Instance(ModelType):
    definition = """
    -> State
    -> Loader
    -> Objective
    -> Optimizer
    -> Scheduler
    parallel        : int unsigned      # parallel group size
    cycle           : int unsigned      # training cycle
    """

    @rowproperty
    def compute(self):
        from foundation.fnn.compute_model import Instance

        return Instance & self


@schema.lookup
class NetworkSetInstance(ModelType):
    definition = """
    -> NetworkSet
    -> State
    -> Loader
    -> Objective
    -> Optimizer
    -> Scheduler
    parallel        : int unsigned      # parallel group size
    cycle           : int unsigned      # training cycle
    """

    @rowproperty
    def compute(self):
        from foundation.fnn.compute_model import NetworkSetInstance

        return NetworkSetInstance & self


# -- Model --


@schema.link
class Model:
    links = [Instance, NetworkSetInstance]
    name = "model"
    comment = "neural network model"


# -- Computed Model --


@schema.computed
class NetworkModel:
    definition = """
    -> Network
    -> Model
    ---
    parameters                              : longblob      # parameter state dict
    network_model_ts = CURRENT_TIMESTAMP    : timestamp     # automatic timestamp
    """

    @property
    def key_source(self):
        keys = [
            Model.Instance,
            Model.NetworkSetInstance * NetworkSet.Member & {"networkset_index": 0},
        ]
        return (Network * Model).proj() & keys

    def make(self, key):
        from foundation.utils.torch import save_to_array

        # network trainer
        trainer = (Model & key).link.compute.train(network_id=key["network_id"])

        for network_id, state_dict in trainer:

            # network parameters
            _key = dict(key, network_id=network_id, parameters=save_to_array(state_dict))

            # insert
            self.insert1(_key)

    @rowmethod
    def parameters(self, device="cpu"):
        """
        Returns
        -------
        dict[str, torch.Tensor]
            pytorch state dict
        """
        from foundation.utils.torch import load_from_array

        # module parameters, mapped to specified device
        return load_from_array(self.fetch1("parameters"), map_location=device)

    @rowproperty
    def model(self):
        """
        Returns
        -------
        fnn.networks.Network
            trained network module
        """
        from foundation.utils.torch import cuda_enabled

        # frozen module
        module = (Network & self).link.module.freeze()

        # module parameters
        device = "cuda" if cuda_enabled() else "cpu"
        module = module.to(device=device)
        params = self.parameters(device=device)

        # load parameters to module
        module.load_state_dict(params)

        return module
