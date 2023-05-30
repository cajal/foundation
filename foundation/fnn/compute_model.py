from datajoint import U
from djutils import keys, merge, rowmethod
from foundation.utils import logger
from foundation.virtual import fnn


# ----------------------------- Model Training -----------------------------


@keys
class _TrainNetwork:
    """Train Network"""

    @property
    def key_list(self):
        return [
            fnn.Network,
            fnn.State,
            fnn.Loader,
            fnn.Objective,
            fnn.Optimizer,
            fnn.Scheduler,
        ]

    @rowmethod
    def train(self, model_id, cycle=0, seed=0, instances=1, rank=0, size=1):
        from torch import device
        from torch.cuda import current_device
        from foundation.fnn.network import Network
        from foundation.fnn.model import Model
        from foundation.fnn.train import State, Scheduler, Optimizer, Loader, Objective
        from foundation.fnn.cache import ModelNetworkInfo, ModelNetworkCheckpoint

        network_id = self.key.fetch1("network_id")
        checkpoint = ModelNetworkCheckpoint & {"model_id": model_id} & "rank >= 0" & f"rank < {size}"
        init = not checkpoint and cycle == 0
        cuda = device("cuda", current_device())

        state = (State & self.key).link.network_state
        module = state.build(network_id, initialize=init).to(device=cuda)

        if checkpoint:
            assert len(checkpoint) == size
            assert len(U("epoch") & checkpoint) == 1

            logger.info("Reloading from previous checkpoint")
            prev = (checkpoint & self.key & {"rank": rank}).load(device=cuda)
            module.load_state_dict(prev["state_dict"])
            optimizer = prev["optimizer"]

        else:
            if cycle == 0:
                logger.info("Initializing training")
            else:
                logger.info("Reloading from previous cycle")
                prev = (Model & {"model_id": model_id}).link.previous_networks
                prev = (prev & self.key).parameters(device=cuda)
                module.load_state_dict(prev)

            scheduler = (Scheduler & self.key).link.scheduler
            scheduler._init(epoch=0, cycle=cycle)

            optimizer = (Optimizer & self.key).link.optimizer
            optimizer._init(scheduler=scheduler)

        data = (Network & self.key).link.data
        dataset = data.link.dataset

        loader = (Loader & self.key).link.loader
        loader._init(dataset=dataset)

        objective = (Objective & self.key).link.objective
        objective._init(module=module)

        params = module.named_parameters()
        groups = module.parallel_groups(instances=instances)

        for epoch, info in optimizer.optimize(
            loader=loader, objective=objective, parameters=params, groups=groups, seed=seed
        ):
            ModelNetworkInfo.fill(
                model_id=model_id,
                network_id=network_id,
                rank=rank,
                epoch=epoch,
                info=info,
            )
            ModelNetworkCheckpoint.fill(
                model_id=model_id,
                network_id=network_id,
                rank=rank,
                epoch=epoch,
                optimizer=optimizer,
                state_dict=module.state_dict(),
            )


@keys
class TrainNetworkSet:
    """Train Network Set"""

    @property
    def key_list(self):
        nets = fnn.NetworkSet.Member * fnn.Network.VisualNetwork
        nets = U("networkset_id").aggr(nets, c="count(distinct core_id)", s="count(distinct streams)")
        return [
            fnn.NetworkSetModel & (nets & "c=1 and s=1").fetch("KEY"),
        ]

    @staticmethod
    def _train(rank, size, model_id, network_id, port=23456, backend="nccl"):
        from torch.cuda import device
        from torch.distributed import init_process_group

        with device(rank):
            init_process_group(
                backend=backend,
                init_method=f"tcp://0.0.0.0:{port}",
                rank=rank,
                world_size=size,
            )
            key = fnn.Network * fnn.Model.NetworkSetModel & {"model_id": model_id, "network_id": network_id}
            cycle, seed, instances = key.fetch1("cycle", "seed", "instances")
            trainer = _TrainNetwork & key
            trainer.train(
                model_id=model_id,
                cycle=cycle,
                seed=seed,
                instances=instances,
                rank=rank,
                size=size,
            )

    @staticmethod
    def _fn(rank, size, port, keys):
        kwargs = keys[rank]
        TrainNetworkSet._train(rank=rank, size=size, port=port, **kwargs)

    @rowmethod
    def train(self):
        from random import randint
        from torch.cuda import device_count
        from torch.multiprocessing import spawn
        from foundation.fnn.train import Scheduler
        from foundation.fnn.network import NetworkSet
        from foundation.fnn.cache import ModelNetworkCheckpoint as Checkpoint

        nets = merge((NetworkSet & self.key).members * self.key, fnn.Model.NetworkSetModel)
        nets = nets.fetch("network_id", "model_id", as_dict=True, order_by="network_id")

        keys = nets * self.key.fetch1("instances")
        size = len(keys)
        assert device_count() >= size

        conn = self.key.connection
        conn.close()
        port = randint(10000, 60000)
        spawn(
            TrainNetworkSet._fn,
            args=(size, port, keys),
            nprocs=size,
            join=True,
        )
        conn.connect()

        key = merge(self.key, fnn.Model.NetworkSetModel)
        epochs = (Scheduler & key).link.epochs
        checkpoints = Checkpoint & nets & "rank >= 0" & f"rank < {size}" & {"epoch": epochs - 1}
        assert len(checkpoints) == size

        keys = U("network_id", "model_id").aggr(checkpoints, rank="min(rank)").fetch(as_dict=True)
        for key in keys:
            yield key["network_id"], (Checkpoint & key).load()["state_dict"]


@keys
class TrainNetwork:
    """Train Network"""

    @property
    def key_list(self):
        return [
            fnn.NetworkModel,
        ]

    @staticmethod
    def _train(rank, size, model_id, port=23456, backend="nccl"):
        from torch.cuda import device
        from torch.distributed import init_process_group

        with device(rank):
            init_process_group(
                backend=backend,
                init_method=f"tcp://0.0.0.0:{port}",
                rank=rank,
                world_size=size,
            )
            key = fnn.Model.NetworkModel & {"model_id": model_id}
            cycle, seed, instances = key.fetch1("cycle", "seed", "instances")
            trainer = _TrainNetwork & key
            trainer.train(
                model_id=model_id,
                cycle=cycle,
                seed=seed,
                instances=instances,
                rank=rank,
                size=size,
            )

    @rowmethod
    def train(self):
        from random import randint
        from torch.cuda import device_count
        from torch.multiprocessing import spawn
        from foundation.fnn.train import Scheduler
        from foundation.fnn.cache import ModelNetworkCheckpoint as Checkpoint

        key = merge(self.key, fnn.Model.NetworkModel)
        size, model_id = key.fetch1("instances", "model_id")
        assert device_count() >= size

        conn = self.key.connection
        conn.close()
        port = randint(10000, 60000)
        spawn(
            TrainNetwork._train,
            args=(size, model_id, port),
            nprocs=size,
            join=True,
        )
        conn.connect()

        epochs = (Scheduler & key).link.epochs
        checkpoints = Checkpoint & key & "rank >= 0" & f"rank < {size}" & {"epoch": epochs - 1}
        assert len(checkpoints) == size

        key = U("network_id", "model_id").aggr(checkpoints, rank="min(rank)").fetch1()
        return key["network_id"], (Checkpoint & key).load()["state_dict"]
