from djutils import keys, rowmethod, rowproperty
from foundation.utils import logger
from foundation.virtual import fnn


# ----------------------------- Instance -----------------------------

# -- Instance Interface --


class InstanceType:
    """Instance"""

    @rowmethod
    def instantiate(self, data_id, network_id):
        """
        Parameters
        ----------
        data_id : str
            key (foundation.fnn.data.Data)
        network_id : str
            key (foundation.fnn.network.Network)

        Yields
        ------
        str
            data_id (foundation.fnn.data.Data)
        str
            network_id (foundation.fnn.network.Network)
        """
        raise NotImplementedError()


# -- Instance Types --


class ParallelCycle(InstanceType):
    """Parallel Cycle"""

    @property
    def instance_type(self):
        """
        Returns
        -------
        fnn.Instance.* (Individual | Foundation)
            model part table
        """
        raise NotImplementedError()

    @rowproperty
    def shared(self):
        """
        Returns
        -------
        Sequence[str] | None
            shared modules
        """
        return

    @rowmethod
    def _instantiate(self, data_id, network_id, main=False):
        """
        Parameters
        ----------
        network_id : str
            key (foundation.fnn.network.Network)
        data_id : str
            key (foundation.fnn.data.Data)
        main : bool
            main rank
        """
        from foundation.fnn.progress import ModelInfo, ModelCheckpoint, ModelLag, ModelDone
        from foundation.fnn.transfer import Transfer, TransferList
        from foundation.fnn.network import Network
        from foundation.fnn.model import Model
        from foundation.fnn.train import Train
        from foundation.fnn.data import Data
        from foundation.utils import torch_rng
        from torch import device

        # key
        instance_id = (self.instance_type & self.item).fetch1("instance_id")
        key = {"data_id": data_id, "network_id": network_id, "instance_id": instance_id}

        # initialize network
        with torch_rng(seed=self.item["seed"]):
            logger.info(f"Initializing parameters with random seed {self.item['seed']}")

            # initial network
            network = (Network & {"network_id": network_id}).link.network(data_id=data_id).to(device="cuda")

            # transfer network
            if (TransferList & self.item).fetch1("members"):
                logger.info("Transferring from other model")

                # transfers
                transfers = (TransferList & self.item).members.fetch(
                    "transfer_id", order_by="transferlist_index", as_dict=True
                )

                # transfer history
                transferlist_id = TransferList.get(transfers[:-1])["transferlist_id"]

                # transfer method
                transfer = (Transfer & transfers[-1]).link.compute.transfer

                # perform transfer
                network = transfer(
                    transferlist_id=transferlist_id,
                    network_id=network_id,
                    network=network,
                )

        if ModelCheckpoint & key:
            logger.info("Reloading from checkpoint")

            # reload parameters
            parameters = (ModelCheckpoint & key).parameters(device="cuda")
            network.load_state_dict(parameters)

            # reload checkpoint
            checkpoint = (ModelCheckpoint & key).checkpoint(device="cuda")

        elif self.item["cycle"]:
            logger.info("Reloading from previous cycle")

            # previous instance
            prev = dict(self.item)
            prev["cycle"] -= 1
            prev = (self.instance_type & prev).fetch1("instance_id")
            prev = {"data_id": data_id, "network_id": network_id, "instance_id": prev}

            # reload parameters
            parameters = (Model & prev).parameters(device="cuda")
            network.load_state_dict(parameters)

            # no checkpoint
            checkpoint = None

        else:
            logger.info("Starting from init")

            # no checkpoint
            checkpoint = None

        # parallel groups
        groups = network.parallel_groups(group_size=self.item["parallel"], shared=self.shared)

        # dataset
        dataset = (Data & {"data_id": data_id}).link.compute.dataset

        # train
        for epoch, info, checkpoint, parameters in (Train & self.item).link.compute.train(
            dataset=dataset,
            network=network,
            groups=groups,
            checkpoint=checkpoint,
            cycle=self.item["cycle"],
        ):
            if main:
                if epoch:
                    # save lag
                    lag = (ModelCheckpoint & key).fetch1()
                    ModelLag.insert1(lag, replace=True)

                # save info
                ModelInfo.fill(dict(key, epoch=epoch, info=info))

                # save checkpoint
                ModelCheckpoint.fill(dict(key, epoch=epoch, checkpoint=checkpoint, parameters=parameters))

        if main:
            # register done
            ModelDone.insert1(key)


@keys
class Individual(ParallelCycle):
    """Individual"""

    @property
    def keys(self):
        return [
            fnn.Individual,
        ]

    @property
    def instance_type(self):
        return fnn.Instance.Individual

    @staticmethod
    def _spawn(rank, size, data_id, network_id, instance_id, port=23456, backend="nccl"):
        """
        Parameters
        ----------
        rank : int
            distributed rank
        size : int
            distributed size
        data_id : str
            key (foundation.fnn.data.Data)
        network_id : str
            key (foundation.fnn.network.Network)
        instance_id : str
            key (foundation.fnn.instance.Instance)
        port : int
            tcp port
        backend : str
            'nccl' | 'mpi' | 'gloo' | 'ucc'
        """
        from torch.cuda import device
        from torch.distributed import init_process_group

        # main rank
        main = rank == 0

        # cuda device
        with device(rank):
            # distributed process group
            init_process_group(
                backend=backend,
                init_method=f"tcp://0.0.0.0:{port}",
                rank=rank,
                world_size=size,
            )

            # model instance
            key = fnn.Instance.Individual & {"instance_id": instance_id}
            instance = Individual & key

            # instantiate model
            instance._instantiate(data_id=data_id, network_id=network_id, main=main)

    @rowmethod
    def instantiate(self, data_id, network_id):
        from random import randint
        from torch.cuda import device_count
        from torch.multiprocessing import spawn

        # tcp port
        port = randint(10000, 60000)

        # parallel group size, instance_id
        parallel, instance_id = (fnn.Instance.Individual & self.item).fetch1("parallel", "instance_id")

        # verify cuda devices
        assert device_count() >= parallel, "Insufficient cuda devices"

        # instantiate with multiprocessing
        conn = self.key.connection
        conn.close()
        spawn(
            Individual._spawn,
            args=(parallel, data_id, network_id, instance_id, port),
            nprocs=parallel,
            join=True,
        )
        conn.connect()

        # yield model
        yield data_id, network_id


@keys
class Foundation(ParallelCycle):
    """Foundation"""

    @property
    def keys(self):
        return [
            fnn.Foundation,
        ]

    @property
    def instance_type(self):
        return fnn.Instance.Foundation

    @rowproperty
    def shared(self):
        from foundation.fnn.network import ModuleSet

        # shared modules
        modules = (ModuleSet & self.item).members

        # module names
        return modules.fetch("module", order_by="moduleset_index").tolist()

    @staticmethod
    def _spawn(rank, size, data_ids, network_id, instance_id, port=23456, backend="nccl"):
        """
        Parameters
        ----------
        rank : int
            distributed rank
        size : int
            distributed size
        data_ids : List[str]
            key (foundation.fnn.data.Data)
        network_id : str
            key (foundation.fnn.network.Network)
        instance_id : str
            key (foundation.fnn.instance.Instance)
        port : int
            tcp port
        backend : str
            'nccl' | 'mpi' | 'gloo' | 'ucc'
        """
        from torch.cuda import device
        from torch.distributed import init_process_group

        # main rank
        parallel = (fnn.Instance.Foundation & {"instance_id": instance_id}).fetch1("parallel")
        main = rank % parallel == 0

        # cuda device
        with device(rank):
            # distributed process group
            init_process_group(
                backend=backend,
                init_method=f"tcp://0.0.0.0:{port}",
                rank=rank,
                world_size=size,
            )

            # model instance
            key = fnn.Instance.Foundation & {"instance_id": instance_id}
            instance = Foundation & key

            # instantiate model
            instance._instantiate(data_id=data_ids[rank], network_id=network_id, main=main)

    @rowmethod
    def instantiate(self, data_id, network_id):
        from random import randint
        from torch.cuda import device_count
        from torch.multiprocessing import spawn
        from foundation.fnn.data import DataSet
        from foundation.fnn.progress import ModelCheckpoint

        # tcp port
        port = randint(10000, 60000)

        # data set
        data_ids = set((DataSet & self.item).members.fetch("data_id"))
        assert data_id in data_ids, "Invalid data_id"

        # parallel group size, instance_id
        parallel, instance_id = (fnn.Instance.Foundation & self.item).fetch1("parallel", "instance_id")
        size = parallel * len(data_ids)

        # verify cuda devices
        assert device_count() >= size, "Insufficient cuda devices"

        # verify checkpoints
        checkpoint = ModelCheckpoint & {"network_id": network_id, "instance_id": instance_id}
        if checkpoint:
            epochs, _data_ids = map(set, checkpoint.fetch("epoch", "data_id"))
            assert data_ids == _data_ids, "Invalid checkpoint data_ids"
            assert len(epochs) == 1, "Invalid checkpoint epochs"

        # instantiate with multiprocessing
        conn = self.key.connection
        conn.close()
        spawn(
            Foundation._spawn,
            args=(size, sorted(data_ids), network_id, instance_id, port),
            nprocs=size,
            join=True,
        )
        conn.connect()

        # yield models
        for data_id in sorted(data_ids):
            yield data_id, network_id
