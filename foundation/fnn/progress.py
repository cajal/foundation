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
    def load(self, device="cpu"):
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
        Returns
        -------
        pandas.DataFrame
            info dataframe
        """
        keys = tqdm(self.fetch("KEY", order_by=self.primary_key))
        return pd.DataFrame([dict(k, **(self & k).load(device=device)) for k in keys])


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

        key["checkpoint"] = torch_save(key["checkpoint"])
        cls.insert1(key, replace=True)

    @rowmethod
    def load(self, device="cpu"):
        """
        Returns
        -------
        deserialized object
            row checkpoint
        """
        from foundation.utils.serialize import torch_load

        return torch_load(self.fetch1("checkpoint"), map_location=device)


# ----------------------------- Network Progress -----------------------------


@schema.lookup
class NetworkInfo(Info):
    definition = """
    -> fnn.Network
    -> fnn.Model
    rank                                    : int unsigned  # training rank
    epoch                                   : int unsigned  # training epoch
    ---
    info                                    : longblob      # training info
    network_info_ts = CURRENT_TIMESTAMP     : timestamp     # automatic timestamp
    """


@schema.lookup
class NetworkCheckpoint(Checkpoint):
    definition = """
    -> fnn.Network
    -> fnn.Model
    rank                                        : int unsigned  # training rank
    ---
    epoch                                       : int unsigned  # training epoch
    checkpoint                                  : longblob      # training checkpoint
    network_checkpoint_ts = CURRENT_TIMESTAMP   : timestamp     # automatic timestamp
    """


@schema.lookup
class NetworkDone:
    definition = """
    -> fnn.Network
    -> fnn.Model
    rank                                    : int unsigned  # training rank
    ---
    epoch                                   : int unsigned  # training epoch
    network_done_ts = CURRENT_TIMESTAMP     : timestamp     # automatic timestamp
    """


# ----------------------------- Visual Network Descent Progress -----------------------------


@schema.lookup
class VisualNetworkDescentInfo(Info):
    definition = """
    -> fnn.NetworkModel
    -> fnn.Descent
    -> fnn.Stimulus
    -> fnn.Optimizer
    -> fnn.Scheduler
    -> fnn.DescentSteps
    -> utility.Resolution
    epoch                                   : int unsigned  # descent epoch
    ---
    info                                    : longblob      # descent info
    descent_info_ts = CURRENT_TIMESTAMP     : timestamp     # automatic timestamp
    """


@schema.lookup
class VisualNetworkDescentCheckpoint(Checkpoint):
    definition = """
    -> fnn.NetworkModel
    -> fnn.Descent
    -> fnn.Stimulus
    -> fnn.Optimizer
    -> fnn.Scheduler
    -> fnn.DescentSteps
    -> utility.Resolution
    ---
    epoch                                       : int unsigned  # descent epoch
    checkpoint                                  : longblob      # descent checkpoint
    descent_checkpoint_ts = CURRENT_TIMESTAMP   : timestamp     # automatic timestamp
    """


@schema.lookup
class VisualNetworkDescentDone:
    definition = """
    -> fnn.NetworkModel
    -> fnn.Descent
    -> fnn.Stimulus
    -> fnn.Optimizer
    -> fnn.Scheduler
    -> fnn.DescentSteps
    -> utility.Resolution
    ---
    epoch                                   : int unsigned  # descent epoch
    descent_done_ts = CURRENT_TIMESTAMP     : timestamp     # automatic timestamp
    """


# ----------------------------- Visual Unit Descent Progress -----------------------------


@schema.lookup
class VisualUnitDescentInfo(Info):
    definition = """
    -> fnn.NetworkModel
    -> fnn.NetworkUnit
    -> fnn.Descent
    -> fnn.Stimulus
    -> fnn.Optimizer
    -> fnn.Scheduler
    -> fnn.DescentSteps
    -> utility.Resolution
    epoch                                   : int unsigned  # descent epoch
    ---
    info                                    : longblob      # descent info
    descent_info_ts = CURRENT_TIMESTAMP     : timestamp     # automatic timestamp
    """


@schema.lookup
class VisualUnitDescentCheckpoint(Checkpoint):
    definition = """
    -> fnn.NetworkModel
    -> fnn.NetworkUnit
    -> fnn.Descent
    -> fnn.Stimulus
    -> fnn.Optimizer
    -> fnn.Scheduler
    -> fnn.DescentSteps
    -> utility.Resolution
    ---
    epoch                                       : int unsigned  # descent epoch
    checkpoint                                  : longblob      # descent checkpoint
    descent_checkpoint_ts = CURRENT_TIMESTAMP   : timestamp     # automatic timestamp
    """


@schema.lookup
class VisualUnitDescentDone:
    definition = """
    -> fnn.NetworkModel
    -> fnn.NetworkUnit
    -> fnn.Descent
    -> fnn.Stimulus
    -> fnn.Optimizer
    -> fnn.Scheduler
    -> fnn.DescentSteps
    -> utility.Resolution
    ---
    epoch                                   : int unsigned  # descent epoch
    descent_done_ts = CURRENT_TIMESTAMP     : timestamp     # automatic timestamp
    """
