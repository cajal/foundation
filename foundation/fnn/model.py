from djutils import rowproperty, rowmethod, merge
from foundation.utils import tqdm
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
    parameters                              : blob@external     # parameter state dict
    network_model_ts = CURRENT_TIMESTAMP    : timestamp         # automatic timestamp
    """

    @property
    def key_source(self):
        keys = [
            Model.Instance,
            Model.NetworkSetInstance * NetworkSet.Member & {"networkset_index": 0},
        ]
        return (Network * Model).proj() & keys

    def make(self, key):
        from foundation.utils.serialize import torch_save

        # network trainer
        trainer = (Model & key).link.compute.train(network_id=key["network_id"])

        for network_id, state_dict in trainer:

            # network parameters
            _key = dict(key, network_id=network_id, parameters=torch_save(state_dict))

            # insert
            self.insert1(_key)

    @rowmethod
    def parameters(self, device="cpu"):
        """
        Parameters
        ----------
        device : "cpu" | "cuda" | torch.device
            device to allocate tensors

        Returns
        -------
        dict[str, torch.Tensor]
            pytorch state dict
        """
        from foundation.utils.serialize import torch_load

        # module parameters, mapped to specified device
        return torch_load(self.fetch1("parameters"), map_location=device)

    @rowproperty
    def model(self):
        """
        Returns
        -------
        fnn.networks.Network
            trained network module
        """
        from foundation.utils.context import cuda_enabled

        # frozen module
        module = (Network & self).link.module.freeze()

        # module parameters
        device = "cuda" if cuda_enabled() else "cpu"
        module = module.to(device=device)
        params = self.parameters(device=device)

        # load parameters to module
        module.load_state_dict(params)

        return module


@schema.computed
class NetworkSetModel:
    definition = """
    -> NetworkSet
    -> Model
    module          : varchar(128)  # module name
    ---
    shared          : bool          # shared parameters
    """

    @property
    def key_source(self):
        keys = (NetworkSet.Member * Model * NetworkModel).proj()
        keys &= [
            Model.NetworkSetInstance,
        ]
        keys = (NetworkSet * Model).aggr(keys, n="count(*)") * NetworkSet & "n=members"
        return keys.proj()

    def make(self, key):
        from torch import allclose

        nets = (NetworkSet & key).members
        nets = merge(nets, NetworkModel & key)
        nets = nets.fetch("KEY", order_by="networkset_index")

        model = (NetworkModel & nets[0]).model
        params = {k: v.state_dict() for k, v in model.named_children()}
        shared = set(params)

        for net in tqdm(nets[1:]):

            _model = (NetworkModel & net).model
            _params = {k: v.state_dict() for k, v in _model.named_children()}
            not_shared = set()

            for module in shared:

                if set(params[module]) != set(_params[module]):
                    not_shared |= {module}
                    continue

                for k, v in params[module].items():
                    _v = _params[module][k]

                    if v.shape != _v.shape or not allclose(v, _v):
                        not_shared |= {module}
                        break

            shared = shared - not_shared

        keys = [dict(key, module=module, shared=True) for module in shared]
        keys += [dict(key, module=module, shared=False) for module in set(params) - shared]
        self.insert(keys)
