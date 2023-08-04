import pandas as pd
from djutils import rowmethod
from foundation.utils import tqdm
from foundation.virtual import utility, fnn
from foundation.schemas import fnn as schema


# ----------------------------- Progress Bases -----------------------------


class Info:
    """Info"""

    @classmethod
    def fill(cls, key):
        """
        Parameters
        ----------
        key : dict[str, ...]
            row to insert
        """
        from foundation.utils.serialize import torch_save

        key["info"] = torch_save(key["info"])

        cls.insert1(key)

    @rowmethod
    def info(self, device="cpu"):
        """
        Returns
        -------
        dict
            row info
        """
        from foundation.utils.serialize import torch_load

        return torch_load(self.fetch1("info"), map_location=device)

    def df(self, device="cpu"):
        """
        Parameters
        ----------
        device : "cpu" | "cuda" | torch.device
            device to allocate tensors

        Returns
        -------
        pandas.DataFrame
            info dataframe
        """
        keys = tqdm(self.fetch("KEY", order_by=self.primary_key))
        return pd.DataFrame([dict(k, **(self & k).info(device=device)) for k in keys])


class Checkpoint:
    """Checkpoint"""

    @classmethod
    def fill(cls, key):
        """
        Parameters
        ----------
        key : dict[str, ...]
            row to insert
        """
        from foundation.utils.serialize import torch_save

        key["optimizer"] = torch_save(key["optimizer"])
        key["parameters"] = torch_save(key["parameters"])

        cls.insert1(key, replace=True)

    @rowmethod
    def optimizer(self, device="cpu"):
        """
        Parameters
        ----------
        device : "cpu" | "cuda" | torch.device
            device to allocate tensors

        Returns
        -------
        fnn.train.optimizers.Optimizer
            optimizer
        """
        from foundation.utils.serialize import torch_load

        return torch_load(self.fetch1("optimizer"), map_location=device)

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
            parameters
        """
        from foundation.utils.serialize import torch_load

        return torch_load(self.fetch1("parameters"), map_location=device)


# ----------------------------- Model Progress -----------------------------


@schema.lookup
class ModelInfo(Info):
    definition = """
    -> fnn.Data
    -> fnn.Network
    -> fnn.Instance
    epoch           : int unsigned      # training epoch
    ---
    info            : blob@external     # training info
    """


@schema.lookup
class ModelCheckpoint(Checkpoint):
    definition = """
    -> fnn.Data
    -> fnn.Network
    -> fnn.Instance
    ---
    epoch           : int unsigned      # training epoch
    optimizer       : blob@external     # fnn optimizer
    parameters      : blob@external     # fnn parameters
    """


@schema.lookup
class ModelDone:
    definition = """
    -> ModelCheckpoint
    """


# # ----------------------------- Visual Network Descent Progress -----------------------------


# @schema.lookup
# class VisualNetworkDescentInfo(Info):
#     definition = """
#     -> fnn.NetworkModel
#     -> fnn.Descent
#     -> fnn.Stimulus
#     -> fnn.Optimizer
#     -> fnn.Scheduler
#     -> fnn.DescentSteps
#     -> utility.Resolution
#     epoch                                       : int unsigned      # descent epoch
#     ---
#     info                                        : blob@external     # descent info
#     descent_info_ts = CURRENT_TIMESTAMP         : timestamp         # automatic timestamp
#     """


# @schema.lookup
# class VisualNetworkDescentCheckpoint(Checkpoint):
#     definition = """
#     -> fnn.NetworkModel
#     -> fnn.Descent
#     -> fnn.Stimulus
#     -> fnn.Optimizer
#     -> fnn.Scheduler
#     -> fnn.DescentSteps
#     -> utility.Resolution
#     ---
#     epoch                                       : int unsigned      # descent epoch
#     checkpoint                                  : blob@external     # descent checkpoint
#     descent_checkpoint_ts = CURRENT_TIMESTAMP   : timestamp         # automatic timestamp
#     """


# @schema.lookup
# class VisualNetworkDescentDone:
#     definition = """
#     -> fnn.NetworkModel
#     -> fnn.Descent
#     -> fnn.Stimulus
#     -> fnn.Optimizer
#     -> fnn.Scheduler
#     -> fnn.DescentSteps
#     -> utility.Resolution
#     ---
#     epoch                                       : int unsigned      # descent epoch
#     descent_done_ts = CURRENT_TIMESTAMP         : timestamp         # automatic timestamp
#     """


# # ----------------------------- Visual Unit Descent Progress -----------------------------


# @schema.lookup
# class VisualUnitDescentInfo(Info):
#     definition = """
#     -> fnn.NetworkModel
#     -> fnn.NetworkUnit
#     -> fnn.Descent
#     -> fnn.Stimulus
#     -> fnn.Optimizer
#     -> fnn.Scheduler
#     -> fnn.DescentSteps
#     -> utility.Resolution
#     epoch                                       : int unsigned      # descent epoch
#     ---
#     info                                        : blob@external     # descent info
#     descent_info_ts = CURRENT_TIMESTAMP         : timestamp         # automatic timestamp
#     """


# @schema.lookup
# class VisualUnitDescentCheckpoint(Checkpoint):
#     definition = """
#     -> fnn.NetworkModel
#     -> fnn.NetworkUnit
#     -> fnn.Descent
#     -> fnn.Stimulus
#     -> fnn.Optimizer
#     -> fnn.Scheduler
#     -> fnn.DescentSteps
#     -> utility.Resolution
#     ---
#     epoch                                           : int unsigned      # descent epoch
#     checkpoint                                      : blob@external     # descent checkpoint
#     descent_checkpoint_ts = CURRENT_TIMESTAMP       : timestamp         # automatic timestamp
#     """


# @schema.lookup
# class VisualUnitDescentDone:
#     definition = """
#     -> fnn.NetworkModel
#     -> fnn.NetworkUnit
#     -> fnn.Descent
#     -> fnn.Stimulus
#     -> fnn.Optimizer
#     -> fnn.Scheduler
#     -> fnn.DescentSteps
#     -> utility.Resolution
#     ---
#     epoch                                           : int unsigned      # descent epoch
#     descent_done_ts = CURRENT_TIMESTAMP             : timestamp         # automatic timestamp
#     """
