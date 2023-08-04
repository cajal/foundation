from djutils import keys, rowmethod
from foundation.utils import logger
from foundation.virtual import fnn


# ----------------------------- Train -----------------------------

# -- Train Interface --


class TrainType:
    """Train"""

    @rowmethod
    def train(self, dataset, network, optimizer=None, groups=None, cycle=0):
        """
        Parameters
        ----------
        dataset : fnn.data.dataset.Dataset
            fnn dataset
        network : fnn.model.networks.Network
            fnn network
        optimizer : None | fnn.train.optimizers.Optimizer
            fnn optimizer
        groups : None | Iterable[fnn.train.parallel.ParameterGroup]
            None | parameter groups
        cycle : int
            training cycle

        Yields
        ------
        int
            epoch
        dict
            info
        fnn.model.networks.Network
            network
        fnn.train.optimizers.Optimizer
            optimizer
        """
        raise NotImplementedError()


# -- Train Types --


@keys
class Optimize(TrainType):
    @property
    def keys(self):
        return [
            fnn.Optimize,
        ]

    @rowmethod
    def train(self, dataset, network, optimizer=None, groups=None, cycle=0):
        from foundation.fnn.train import Optimizer, Scheduler, Loader, Objective

        if optimizer is None:
            # scheduler
            scheduler = (Scheduler & self.item).link.scheduler
            scheduler._init(epoch=0, cycle=cycle)

            # optimizer
            optimizer = (Optimizer & self.item).link.optimizer
            optimizer._init(scheduler=scheduler)

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
            yield epoch, info, network, optimizer
