from djutils import rowproperty
from foundation.schemas import fnn as schema


# -------------- Loader --------------

# -- Loader Base --


class _Loader:
    """Loader"""

    @rowproperty
    def loader(self):
        """
        Returns
        -------
        fnn.train.loaders.Loader
            data loader
        """
        raise NotImplementedError()


# -- Loader Types --


@schema.lookup
class RandomBatches(_Loader):
    definition = """
    sample_size     : int unsigned  # samples in a datapoint
    batch_size      : int unsigned  # datapoints in a batch
    epoch_size      : int unsigned  # batches in an epoch
    train_fraction  : decimal(6, 6) # training fraction
    split_seed      : int unsigned  # random seed for train/val splitting
    """

    @rowproperty
    def loader(self):
        from fnn.train.loaders import RandomBatches

        return RandomBatches(**self.fetch1())


# -- Loader --


@schema.link
class Loader:
    links = [RandomBatches]
    name = "loader"


# -------------- Objective --------------

# -- Objective Base --


class _Objective:
    """Objective"""

    @rowproperty
    def objective(self):
        """
        Returns
        -------
        fnn.train.objectives.Objective
            training objective
        """
        raise NotImplementedError()


# -- Objective Types --


@schema.lookup
class ArchitectureLoss(_Objective):
    definition = """
    sample_stream   : bool          # sample stream during training
    burnin_frames   : int unsigned  # initial losses discarded
    """

    @rowproperty
    def objective(self):
        from fnn.train.objectives import ArchitectureLoss

        return ArchitectureLoss(**self.fetch1())


# -- Objective --


@schema.link
class Objective:
    links = [ArchitectureLoss]
    name = "objective"


# -------------- Optimizer --------------

# -- Optimizer Base --


class _Optimizer:
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
class SgdClip(_Optimizer):
    definition = """
    lr          : decimal(9, 6)     # learning rate
    decay       : decimal(9, 6)     # weight decay
    momentum    : decimal(6, 6)     # momentum factor
    nesterov    : bool              # enables nesterov momentum
    clip        : decimal(6, 6)     # adaptive gradient clipping factor
    eps         : decimal(6, 6)     # adaptive gradient clipping mininum
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


# -------------- Scheduler --------------

# -- Scheduler Base --


class _Scheduler:
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
class CosineLr(_Scheduler):
    definition = """
    cycle_size      : int unsigned  # epochs in a cycle
    burnin_epochs   : int unsigned  # burnin epochs
    burnin_cycles   : int unsigned  # burnin cycles
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


# -------------- Trainer --------------

# -- Trainer Base --


class _Trainer:
    """Trainer"""

    # @rowproperty
    # def objective(self):
    #     """
    #     Returns
    #     -------
    #     fnn.train.objectives.Objective
    #         training objective
    #     """
    #     raise NotImplementedError()


# -- Trainer Types --


@schema.lookup
class Parallel(_Trainer):
    definition = """
    -> Loader
    -> Objective
    -> Optimizer
    -> Scheduler
    duplicates      : int unsigned  # training duplicates
    """

    # @rowproperty
    # def objective(self):
    #     from fnn.train.objectives import ArchitectureLoss

    #     return ArchitectureLoss(**self.fetch1())


# -- Trainer --


@schema.link
class Trainer:
    links = [Parallel]
    name = "trainer"