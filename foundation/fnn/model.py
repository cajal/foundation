from djutils import rowproperty, rowmethod
from foundation.fnn.data import Data
from foundation.fnn.network import Network
from foundation.fnn.instance import Instance
from foundation.fnn.progress import ModelDone
from foundation.schemas import fnn as schema


# ----------------------------- Model -----------------------------


@schema.computed
class Model:
    definition = """
    -> Data
    -> Network
    -> Instance
    ---
    -> ModelDone
    """

    @property
    def key_source(self):
        return Data.proj() * Network.proj() * Instance.proj()

    def make(self, key):
        # data
        data_id = key.pop("data_id")

        # network
        network_id = key.pop("network_id")

        # instance
        instance = (Instance & key).link.compute

        # instantiate
        for data_id, network_id in instance.instantiate(data_id=data_id, network_id=network_id):

            # insert
            self.insert1(dict(key, data_id=data_id, network_id=network_id))

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
        raise NotImplementedError()

    @rowproperty
    def model(self):
        """
        Returns
        -------
        fnn.networks.Network
            trained network model
        """
        raise NotImplementedError()
