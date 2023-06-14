import numpy as np
from djutils import keys, merge, rowmethod
from foundation.utils import logger
from foundation.virtual import fnn


# ----------------------------- Network Model -----------------------------

# -- Network Model Base --


class NetworkModel:
    """Network Model"""

    @rowmethod
    def train(self, network_id):
        """
        Parameters
        ----------
        network_id : str
            key (foundation.fnn.network.Network)

        Yields
        ------
        str
            network_id
        dict
            network parameters (pytorch state dict)
        """
        raise NotImplementedError()


# -- Network Model Types --


class _Instance(NetworkModel):
    """Network Instance Base"""

    @property
    def model_type(self):
        """
        Returns
        -------
        foundation.fnn.Model.* (Instance|NetworkSetInstance)
            model part table
        """
        raise NotImplementedError()

    @rowmethod
    def _train(self, rank, network_id):
        """
        Parameters
        ----------
        rank : int
            distributed rank
        network_id : str
            key (foundation.fnn.network.Network)
        """
        from torch import device
        from torch.cuda import current_device
        from foundation.fnn.network import Network
        from foundation.fnn.model import Model, NetworkModel
        from foundation.fnn.train import State, Scheduler, Optimizer, Loader, Objective
        from foundation.fnn.cache import NetworkInfo, NetworkCheckpoint, NetworkDone

        # model id, cycle, checkpoint
        key = merge(self.key, self.model_type)
        model_id, cycle, groups = key.fetch1("model_id", "cycle", "parallel")
        checkpoint = NetworkCheckpoint & {"rank": rank, "network_id": network_id, "model_id": model_id}
        initialize = (cycle == 0) and (not checkpoint)

        # network module
        module = (State & self.key).link.state.build(network_id=network_id, initialize=initialize)
        cuda = device("cuda", current_device())
        module = module.to(device=cuda)

        if checkpoint:
            logger.info("Reloading from checkpoint")

            # load checkpoint
            prev = checkpoint.load(device=cuda)

            # reload parameters
            module.load_state_dict(prev["state_dict"])

            # optimizer
            optimizer = prev["optimizer"]

        else:
            if cycle > 0:
                logger.info("Reloading from previous cycle")

                # previous model id
                key = self.key.fetch1()
                key["cycle"] -= 1
                _model_id = (self.model_type & key).fetch1("model_id")

                # load previous cycle
                network = NetworkModel & {"network_id": network_id, "model_id": _model_id}
                params = network.parameters(device="cuda")

                # reload parameters
                module.load_state_dict(params)

            # scheduler
            scheduler = (Scheduler & self.key).link.scheduler
            scheduler._init(epoch=0, cycle=cycle)

            # optimizer
            optimizer = (Optimizer & self.key).link.optimizer
            optimizer._init(scheduler=scheduler)

        # training objective
        objective = (Objective & self.key).link.objective
        objective._init(module=module)

        # training dataset
        network = Network & {"network_id": network_id}
        dataset = network.link.data.trainset

        # data loader
        loader = (Loader & self.key).link.loader
        loader._init(dataset=dataset)

        # parameters and parallel groups
        params = module.named_parameters()
        groups = module.parallel_groups(groups=groups)

        # train epochs
        for epoch, info in optimizer.optimize(loader=loader, objective=objective, parameters=params, groups=groups):

            # log epoch info
            NetworkInfo.fill(
                network_id=network_id,
                model_id=model_id,
                rank=rank,
                epoch=epoch,
                info=info,
            )

            # save checkpoint
            NetworkCheckpoint.fill(
                network_id=network_id,
                model_id=model_id,
                rank=rank,
                epoch=epoch,
                checkpoint={"optimizer": optimizer, "state_dict": module.state_dict()},
            )

        # register done
        key = {"rank": rank, "network_id": network_id, "model_id": model_id, "epoch": epoch}
        NetworkDone.insert1(key)


@keys
class Instance(_Instance):
    """Network Instance"""

    @property
    def key_list(self):
        return [
            fnn.Instance,
        ]

    @property
    def model_type(self):
        return fnn.Model.Instance

    @staticmethod
    def _fn(rank, size, model_id, network_id, port=23456, backend="nccl"):
        """
        Parameters
        ----------
        rank : int
            distributed rank
        size : int
            distributed size
        model_id : str
            key (foundation.fnn.model.Model)
        network_id : str
            key (foundation.fnn.network.Network)
        port : int
            tcp port
        backend : str
            'nccl' | 'mpi' | 'gloo' | 'ucc'
        """
        from torch.cuda import device
        from torch.distributed import init_process_group

        # cuda device
        with device(rank):

            # distributed process group
            init_process_group(
                backend=backend,
                init_method=f"tcp://0.0.0.0:{port}",
                rank=rank,
                world_size=size,
            )

            # network instance trainer
            key = fnn.Model.Instance & {"model_id": model_id}
            trainer = Instance & key

            # train network
            trainer._train(rank=rank, network_id=network_id)

    @rowmethod
    def train(self, network_id):
        from random import randint
        from torch.cuda import device_count
        from torch.multiprocessing import spawn
        from foundation.fnn.cache import NetworkCheckpoint, NetworkDone

        # tcp port
        port = randint(10000, 60000)

        # parallel groups, model_id
        size, model_id = merge(self.key, fnn.Model.Instance).fetch1("parallel", "model_id")

        # verify cuda devices
        assert device_count() >= size, "Insufficient cuda devices"

        # verify checkpoints
        checkpoint = NetworkCheckpoint & {"model_id": model_id, "network_id": network_id}
        if checkpoint:
            ranks, epochs = checkpoint.fetch("rank", "epoch", order_by="rank")
            assert np.array_equal(ranks, np.arange(size)), "Invalid checkpoint ranks"
            assert np.unique(epochs).size == 1, "Invalid checkpoint epochs"

        # train with multiprocessing
        conn = self.key.connection
        conn.close()
        spawn(
            Instance._fn,
            args=(size, model_id, network_id, port),
            nprocs=size,
            join=True,
        )
        conn.connect()

        # trained keys
        done = NetworkDone & {"model_id": model_id, "network_id": network_id}
        done = done.fetch(as_dict=True, order_by="rank")

        # verify keys
        assert np.array_equal([_["rank"] for _ in done], np.arange(size)), "Invalid done ranks"
        assert np.unique([_["epoch"] for _ in done]).size == 1, "Invalid done epochs"

        # trained network parameters
        params = (NetworkCheckpoint & done[0]).load()["state_dict"]

        yield network_id, params


@keys
class NetworkSetInstance(_Instance):
    """Network Set Instance"""

    @property
    def key_list(self):
        return [
            fnn.ParallelNetworkSet,
            fnn.NetworkSetInstance,
        ]

    @property
    def model_type(self):
        return fnn.Model.NetworkSetInstance

    @staticmethod
    def _fn(rank, size, model_id, network_ids, port=23456, backend="nccl"):
        """
        Parameters
        ----------
        rank : int
            distributed rank
        size : int
            distributed size
        model_id : str
            key (foundation.fnn.model.Model)
        network_ids : List[str]
            list of keys (foundation.fnn.network.Network)
        port : int
            tcp port
        backend : str
            'nccl' | 'mpi' | 'gloo' | 'ucc'
        """
        from torch.cuda import device
        from torch.distributed import init_process_group

        # cuda device
        with device(rank):

            # distributed process group
            init_process_group(
                backend=backend,
                init_method=f"tcp://0.0.0.0:{port}",
                rank=rank,
                world_size=size,
            )

            # network instance trainer
            key = fnn.Model.NetworkSetInstance & {"model_id": model_id}
            trainer = NetworkSetInstance & key

            # train network
            trainer._train(rank=rank, network_id=network_ids[rank])

    @rowmethod
    def train(self, network_id):
        from random import randint
        from torch.cuda import device_count
        from torch.multiprocessing import spawn
        from foundation.fnn.network import NetworkSet
        from foundation.fnn.cache import NetworkCheckpoint, NetworkDone

        # tcp port
        port = randint(10000, 60000)

        # network set
        networks = (NetworkSet & self.key).members.fetch("network_id", order_by="networkset_index")
        assert network_id in networks, "Invalid network_id"

        # parallel groups, model_id
        parallel, model_id = merge(self.key, fnn.Model.NetworkSetInstance).fetch1("parallel", "model_id")

        # parallel networks
        parallel_networks = np.repeat(networks, parallel).tolist()
        size = len(parallel_networks)

        # verify cuda devices
        assert device_count() >= size, "Insufficient cuda devices"

        # verify checkpoints
        checkpoint = NetworkCheckpoint & {"model_id": model_id}
        if checkpoint:
            network_ids, ranks, epochs = checkpoint.fetch("network_id" "rank", "epoch", order_by="rank")
            assert np.array_equal(network_ids, parallel_networks), "Invalid checkpoint network_ids"
            assert np.array_equal(ranks, np.arange(size)), "Invalid checkpoint ranks"
            assert np.unique(epochs).size == 1, "Invalid checkpoint epochs"

        # train with multiprocessing
        conn = self.key.connection
        conn.close()
        spawn(
            NetworkSetInstance._fn,
            args=(size, model_id, parallel_networks, port),
            nprocs=size,
            join=True,
        )
        conn.connect()

        # trained keys
        done = NetworkDone & {"model_id": model_id}
        done = done.fetch(as_dict=True, order_by="rank")

        # verify keys
        assert np.array_equal([_["network_id"] for _ in done], parallel_networks), "Invalid done network_ids"
        assert np.array_equal([_["rank"] for _ in done], np.arange(size)), "Invalid done ranks"
        assert np.unique([_["epoch"] for _ in done]).size == 1, "Invalid done epochs"

        # trained network parameters
        for i, network_id in enumerate(networks):

            key = done[i * parallel]
            params = (NetworkCheckpoint & key).load()["state_dict"]

            yield network_id, params
