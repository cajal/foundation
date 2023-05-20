from djutils import rowmethod, Filepath
from foundation.fnn.network import Network
from foundation.fnn.model import Model
from foundation.schemas import fnn as schema


@schema.lookup
class NetworkModelInfo(Filepath):
    definition = """
    -> Network
    -> Model
    rank        : int unsigned          # training rank
    epoch       : int unsigned          # training epoch
    ---
    info        : filepath@scratch09    # training info
    """

    @classmethod
    def fill(cls, network_id, model_id, rank, epoch, info):
        from torch import save

        key = dict(network_id=network_id, model_id=model_id, rank=rank, epoch=epoch)

        filepath = cls.createpath(key, "info", "pkl")
        save(info, filepath, pickle_protocol=5)

        cls.insert1(dict(key, info=filepath))

    @rowmethod
    def load(self, device="cpu"):
        from torch import load

        filepath = self.fetch1("info")
        return load(filepath, map_location=device)


@schema.lookup
class NetworkModelCheckpoint(Filepath):
    definition = """
    -> Network
    -> Model
    rank        : int unsigned          # training rank
    ---
    epoch       : int unsigned          # training epoch
    checkpoint  : filepath@scratch09    # training checkpoint
    """

    @classmethod
    def fill(cls, network_id, model_id, rank, epoch, optimizer, state_dict):
        from torch import save

        key = dict(network_id=network_id, model_id=model_id, rank=rank)

        if epoch > 0:
            assert (cls & key).fetch1("epoch") == epoch - 1

        filepath = cls.createpath(key, "checkpoint", "pkl")
        save({"optimizer": optimizer, "state_dict": state_dict}, filepath, pickle_protocol=5)

        row = dict(key, epoch=epoch, checkpoint=filepath)
        if epoch > 0:
            (cls & key).replace(row, prompt=False)
        else:
            cls.insert1(row)

    @rowmethod
    def load(self, device="cpu"):
        from torch import load

        filepath = self.fetch1("info")
        return load(filepath, map_location=device)
