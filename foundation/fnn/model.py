from djutils import rowmethod
from foundation.fnn.data import Data, DataSet
from foundation.fnn.network import Network
from foundation.fnn.instance import Instance
from foundation.fnn.progress import ModelDone, ModelCheckpoint
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
        key = [
            Instance.Individual,
            Instance.Foundation * DataSet.Member & "dataset_index=0",
        ]
        return (Data * Network * Instance).proj() & key

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
            network parameters
        """
        return (ModelCheckpoint & self).parameters(device)

    @rowmethod
    def model(self, device="cpu"):
        """
        Parameters
        ----------
        device : "cpu" | "cuda" | torch.device
            device to allocate tensors

        Returns
        -------
        fnn.networks.Network
            trained network
        """
        # load network
        net = (Network & self).link.network(data_id=self.fetch1("data_id"))
        net = net.to(device=device).freeze(True)

        # load parameters
        params = self.parameters(device=device)
        net.load_state_dict(params)

        return net
