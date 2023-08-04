from djutils import rowproperty
from foundation.schemas import fnn as schema


# ----------------------------- Optimizer -----------------------------

# -- Optimizer Interface --


class OptimizerType:
    """Optimizer"""

    @rowproperty
    def optimizer(self):
        """
        Returns
        -------
        fnn.train.optimizers.Optimizer
            module optimizer
        """
        raise NotImplementedError()


# -- Optimizer Types --


@schema.lookup
class SgdClip(OptimizerType):
    definition = """
    lr          : decimal(9, 6)     # learning rate
    decay       : decimal(9, 6)     # weight decay
    momentum    : decimal(6, 6)     # momentum factor
    nesterov    : bool              # enables nesterov momentum
    clip        : decimal(6, 6)     # adaptive gradient clipping factor
    eps         : decimal(6, 6)     # adaptive gradient clipping mininum
    seed        : int unsigned      # seed for optimization
    """

    @rowproperty
    def optimizer(self):
        from fnn.train.optimizers import SgdClip

        return SgdClip(**self.fetch1())


# -- Optimizer --


@schema.link
class Optimizer:
    links = [SgdClip]
    name = "optimizer"
    comment = "module optimizer"


# ----------------------------- Scheduler -----------------------------

# -- Scheduler Interface --


class SchedulerType:
    """Scheduler"""

    @rowproperty
    def scheduler(self):
        """
        Returns
        -------
        fnn.train.schedulers.Scheduler
            hyperparameter scheduler
        """
        raise NotImplementedError()


# -- Scheduler Types --


@schema.lookup
class CosineLr(SchedulerType):
    definition = """
    cycle_size      : int unsigned  # epochs in a cycle
    warmup_epochs   : int unsigned  # burnin epochs
    warmup_cycles   : int unsigned  # burnin cycles
    """

    @rowproperty
    def scheduler(self):
        from fnn.train.schedulers import CosineLr

        return CosineLr(**self.fetch1())


# -- Scheduler --


@schema.link
class Scheduler:
    links = [CosineLr]
    name = "scheduler"
    comment = "hyperparameter scheduler"


# ----------------------------- Loader -----------------------------

# -- Loader Interface --


class LoaderType:
    """Loader"""

    @rowproperty
    def loader(self):
        """
        Returns
        -------
        fnn.train.loaders.DatasetLoader
            data loader
        """
        raise NotImplementedError()


# -- Loader Types --


@schema.lookup
class Batches(LoaderType):
    definition = """
    sample_size     : int unsigned  # samples in a datapoint
    batch_size      : int unsigned  # datapoints in a batch
    training_size   : int unsigned  # training batches in an epoch
    validation_size : int unsigned  # validation batches in an epoch
    """

    @rowproperty
    def loader(self):
        from fnn.train.loaders import Batches

        return Batches(**self.fetch1())


# -- Loader --


@schema.link
class Loader:
    links = [Batches]
    name = "loader"
    comment = "data loader"


# ----------------------------- Objective -----------------------------

# -- Objective Interface --


class ObjectiveType:
    """Network Objective"""

    @rowproperty
    def objective(self):
        """
        Returns
        -------
        fnn.train.objectives.NetworkObjective
            network objective
        """
        raise NotImplementedError()


# -- Objective Types --


@schema.lookup
class NetworkLoss(ObjectiveType):
    definition = """
    sample_stream   : bool          # sample stream during training
    burnin_frames   : int unsigned  # initial losses discarded
    """

    @rowproperty
    def objective(self):
        from fnn.train.objectives import NetworkLoss

        return NetworkLoss(**self.fetch1())


# -- Objective --


@schema.link
class Objective:
    links = [NetworkLoss]
    name = "objective"
    comment = "training objective"


# ----------------------------- Train -----------------------------

# -- Train Interface --


class TrainType:
    """Train"""

    @rowproperty
    def compute(self):
        """
        Returns
        -------
        foundation.fnn.compute_train.TrainType (row)
            compute train
        """
        raise NotImplementedError()


# -- Train Types --


@schema.lookup
class Optimize(TrainType):
    definition = """
    -> Optimizer
    -> Scheduler
    -> Loader
    -> Objective
    """

    @rowproperty
    def compute(self):
        from foundation.fnn.compute_train import Optimize

        return Optimize & self


# -- Train --


@schema.link
class Train:
    links = [Optimize]
    name = "train"
    comment = "fnn training"
