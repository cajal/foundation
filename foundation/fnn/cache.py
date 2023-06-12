import pandas as pd
from datajoint import U
from djutils import rowmethod
from foundation.utils import tqdm
from foundation.fnn.network import Network
from foundation.fnn.model import Model
from foundation.schemas import fnn as schema


@schema.lookup
class NetworkModelInfo:
    definition = """
    -> Network
    -> Model
    rank        : int unsigned      # training rank
    epoch       : int unsigned      # training epoch
    ---
    info        : longblob          # training info
    """

    @classmethod
    def fill(cls, model_id, network_id, rank, epoch, info):
        from foundation.utils.torch import save_to_array

        key = dict(
            model_id=model_id,
            network_id=network_id,
            rank=rank,
            epoch=epoch,
            info=save_to_array(info),
        )
        cls.insert1(key)

    @rowmethod
    def load(self, device="cpu"):
        from foundation.utils.torch import load_from_array

        return load_from_array(self.fetch1("info"), map_location=device)

    def df(self, device="cpu"):
        keys = tqdm(self.fetch("KEY", order_by=self.primary_key))
        return pd.DataFrame([dict(k, **(NetworkModelInfo & k).load(device=device)) for k in keys])


@schema.lookup
class NetworkModelCheckpoint:
    definition = """
    -> Network
    -> Model
    rank        : int unsigned      # training rank
    ---
    epoch       : int unsigned      # training epoch
    checkpoint  : longblob          # training checkpoint
    """

    @classmethod
    def fill(cls, model_id, network_id, rank, epoch, optimizer, state_dict):
        from foundation.utils.torch import save_to_array

        key = dict(
            model_id=model_id,
            network_id=network_id,
            rank=rank,
            epoch=epoch,
            checkpoint=save_to_array({"optimizer": optimizer, "state_dict": state_dict}),
        )
        cls.insert1(key, replace=True)

    @rowmethod
    def load(self, device="cpu"):
        from foundation.utils.torch import load_from_array

        return load_from_array(self.fetch1("checkpoint"), map_location=device)


@schema.lookup
class NetworkModelDone:
    definition = """
    -> Network
    -> Model
    rank        : int unsigned      # training rank
    ---
    epoch       : int unsigned      # training epoch
    """
