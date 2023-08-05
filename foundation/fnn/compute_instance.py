from djutils import keys, rowmethod
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

    @property
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
        from foundation.fnn.progress import ModelInfo, ModelCheckpoint, ModelDone
        from foundation.fnn.transfer import Transfer, TransferList
        from foundation.fnn.network import Network
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

        if ModelCheckpoint & key:
            logger.info("Reloading from checkpoint")

            # reload parameters
            parameters = (ModelCheckpoint & key).parameters(device="cuda")
            network.load_state_dict(parameters)

            # reload optimizer
            optimizer = (ModelCheckpoint & key).optimizer(device="cuda")

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

            # new optimizer
            optimizer = None

        elif (TransferList & self.item).fetch1("members"):
            logger.info("Transferring from other model")

            raise NotImplementedError()  # TODO

            # new optimizer
            optimizer = None

        else:
            logger.info("Starting from init")

            # new optimizer
            optimizer = None

        # dataset
        dataset = (Data & {"data_id": data_id}).link.compute.dataset

        # parallel groups
        groups = network.parallel_groups(group_size=self.item["parallel"])

        # train
        for epoch, info, network, optimizer in (Train & self.item).link.compute.train(
            network=network,
            optimizer=optimizer,
            dataset=dataset,
            groups=groups,
            cycle=self.item["cycle"],
        ):
            if main:
                # save info
                ModelInfo.fill(dict(key, epoch=epoch, info=info))

                # save checkpoint
                parameters = network.state_dict()
                ModelCheckpoint.fill(dict(key, epoch=epoch, optimizer=optimizer, parameters=parameters))

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
        network_id : str
            key (foundation.fnn.network.Network)
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
