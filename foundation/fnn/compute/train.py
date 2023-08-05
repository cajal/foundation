from djutils import keys, rowmethod
from foundation.utils import logger
from foundation.virtual import fnn


# ----------------------------- Train -----------------------------

# -- Train Interface --


class TrainType:
    """Train"""

    @rowmethod
    def train(self, dataset, network, groups=None, checkpoint=None, cycle=0):
        """
        Parameters
        ----------
        dataset : fnn.data.dataset.Dataset
            fnn dataset
        network : fnn.model.networks.Network
            fnn network
        groups : Iterable[fnn.train.parallel.ParameterGroup] | None
            parallel groups
        checkpoint : dict[str, Serializable] | None
            training checkpoint
        cycle : int
            training cycle

        Yields
        ------
        int
            training epoch
        dict[str, Serializable]
            training info
        dict[str, Serializable]
            training checkpoint
        dict[str, torch.Tensor]
            model parameters
        """
        raise NotImplementedError()


# -- Train Types --


@keys
class Optimize(TrainType):
    """Optimize"""

    @property
    def keys(self):
        return [
            fnn.Optimize,
        ]

    @rowmethod
    def train(self, dataset, network, groups=None, checkpoint=None, cycle=0):
        from foundation.fnn.train import Optimizer, Scheduler, Loader, Objective

        if checkpoint is None:
            # scheduler
            scheduler = (Scheduler & self.item).link.scheduler
            scheduler._init(epoch=0, cycle=cycle)

            # optimizer
            optimizer = (Optimizer & self.item).link.optimizer
            optimizer._init(scheduler=scheduler)

        else:
            # optimizer
            optimizer = checkpoint["optimizer"]

        # data loader
        loader = (Loader & self.item).link.loader
        loader._init(dataset=dataset)

        # training objective
        objective = (Objective & self.item).link.objective
        objective._init(network=network)

        # train network
        for epoch, info in optimizer.optimize(
            loader=loader,
            objective=objective,
            parameters=network.named_parameters(),
            groups=groups,
        ):
            yield epoch, info, {"optimizer": optimizer}, network.state_dict()
