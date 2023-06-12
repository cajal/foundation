from djutils import rowproperty, rowmethod
from foundation.fnn.network import Network, NetworkSet
from foundation.fnn.train import State, Loader, Objective, Optimizer, Scheduler
from foundation.schemas import fnn as schema


# ----------------------------- Model -----------------------------

# -- Model Base --


class _Model:
    """Model"""

    @rowproperty
    def model(self):
        """
        Returns
        -------
        foundation.fnn.compute_model.NetworkModel (row)
            network model
        """
        raise NotImplementedError()


# -- Model Types --


@schema.lookup
class Instance(_Model):
    definition = """
    -> State
    -> Loader
    -> Objective
    -> Optimizer
    -> Scheduler
    parallel        : int unsigned      # parallel groups
    cycle           : int unsigned      # training cycle
    """


# -- Model --


@schema.link
class Model:
    links = [Instance]
    name = "model"
    comment = "neural network model"


# -- Computed Model --


@schema.computed
class NetworkModel:
    definition = """
    -> Network
    -> Model
    ---
    parameters      : longblob      # parameter state dict
    """

    @property
    def key_source(self):
        keys = [
            Model.Instance,
        ]
        return (Network * Model).proj() & keys

    def make(self, key):
        from foundation.utils.torch import save_to_array

        # network trainer
        trainer = (Model & key).link.train(network_id=key["network_id"])

        for network_id, state_dict in trainer:

            # network parameters
            _key = dict(key, network_id=network_id, parameters=save_to_array(state_dict))

            # insert
            self.insert1(_key)

    @rowmethod
    def parameters(self, device="cpu"):
        from foundation.utils.torch import load_from_array

        return load_from_array(self.fetch1("parameters"), map_location=device)

    @rowproperty
    def model(self):
        from foundation.utils.torch import cuda_enabled

        module = (Network & self).link.module.freeze()

        device = "cuda" if cuda_enabled() else "cpu"
        module = module.to(device=device)
        params = self.parameters(device=device)

        module.load_state_dict(params)
        return module
