import pandas as pd
from datajoint import U
from djutils import rowmethod
from foundation.utils import tqdm
from foundation.fnn.network import Network
from foundation.fnn.model import Model
from foundation.schemas import fnn as schema


@schema.lookup
class NetworkInfo:
    definition = """
    -> Network
    -> Model
    rank                                    : int unsigned  # training rank
    epoch                                   : int unsigned  # training epoch
    ---
    info                                    : longblob      # training info
    network_info_ts = CURRENT_TIMESTAMP     : timestamp     # automatic timestamp
    """

    @classmethod
    def fill(cls, network_id, model_id, rank, epoch, info):
        """
        Parameters
        ----------
        network_id : str
            key (foundation.fnn.network.Network)
        model_id : str
            key (foundation.fnn.model.Model)
        rank : int
            training rank
        epoch : int
            training epoch
        info : dict
            training info
        """
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
        """
        Returns
        -------
        dict
            training info
        """
        from foundation.utils.torch import load_from_array

        return load_from_array(self.fetch1("info"), map_location=device)

    def df(self, device="cpu"):
        """
        Returns
        -------
        pandas.DataFrame
            training info dataframe
        """
        keys = tqdm(self.fetch("KEY", order_by=self.primary_key))
        return pd.DataFrame([dict(k, **(NetworkInfo & k).load(device=device)) for k in keys])


@schema.lookup
class NetworkCheckpoint:
    definition = """
    -> Network
    -> Model
    rank                                        : int unsigned  # training rank
    ---
    epoch                                       : int unsigned  # training epoch
    checkpoint                                  : longblob      # training checkpoint
    network_checkpoint_ts = CURRENT_TIMESTAMP   : timestamp     # automatic timestamp
    """

    @classmethod
    def fill(cls, network_id, model_id, rank, epoch, checkpoint):
        """
        Parameters
        ----------
        network_id : str
            key (foundation.fnn.network.Network)
        model_id : str
            key (foundation.fnn.model.Model)
        rank : int
            training rank
        epoch : int
            training epoch
        checkpoint : serializable object
            training checkpoint
        """
        from foundation.utils.torch import save_to_array

        key = dict(
            model_id=model_id,
            network_id=network_id,
            rank=rank,
            epoch=epoch,
            checkpoint=save_to_array(checkpoint),
        )
        cls.insert1(key, replace=True)

    @rowmethod
    def load(self, device="cpu"):
        """
        Returns
        -------
        deserialized object
            training checkpoint
        """
        from foundation.utils.torch import load_from_array

        return load_from_array(self.fetch1("checkpoint"), map_location=device)


@schema.lookup
class NetworkDone:
    definition = """
    -> Network
    -> Model
    rank                                    : int unsigned  # training rank
    ---
    epoch                                   : int unsigned  # training epoch
    network_done_ts = CURRENT_TIMESTAMP     : timestamp     # automatic timestamp
    """
