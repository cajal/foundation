from djutils import rowproperty
from foundation.schemas import fnn as schema


# ----------------------------- Feedforward -----------------------------

# -- Feedforward Base --


class _Feedforward:
    """Feedforward"""

    @rowproperty
    def nn(self):
        """
        Returns
        -------
        fnn.model.feedforwards.Feedforward
            feedforward component
        """
        raise NotImplementedError()


# -- Feedforward Types --


@schema.lookup
class Dense(_Feedforward):
    definition = """
    pre_kernel      : int unsigned  # pre kernel size
    pre_stride      : int unsigned  # pre stride
    block_channels  : varchar(128)  # block channels per stream (csv)
    block_groups    : varchar(128)  # block groups per stream (csv)
    block_layers    : varchar(128)  # block layers (csv)
    block_pools     : varchar(128)  # block pool sizes (csv)
    block_kernels   : varchar(128)  # block kernel sizes (csv)
    block_dynamics  : varchar(128)  # block dynamic sizes (csv)
    nonlinear       : varchar(128)  # nonlinearity
    dropout         : decimal(6, 6) # dropout probability
    """

    @rowproperty
    def nn(self):
        from fnn.model.feedforwards import Dense

        kwargs = self.fetch1("KEY")
        for k, v in kwargs.items():
            if k.startswith("block_"):
                kwargs[k] = v.split(",")

        return Dense(**kwargs)


# -- Feedforward --


@schema.link
class Feedforward:
    links = [Dense]
    name = "feedforward"


# ----------------------------- Recurrent -----------------------------

# -- Recurrent Base --


class _Recurrent:
    """Recurrent"""

    @rowproperty
    def nn(self):
        """
        Returns
        -------
        fnn.model.recurrents.Recurrent
            recurrent component
        """
        raise NotImplementedError()


# -- Recurrent Types --


@schema.lookup
class Rvt(_Recurrent):
    definition = """
    channels        : int unsigned  # channels per stream
    groups          : int unsigned  # groups per stream
    kernel_size     : int unsigned  # kernel size
    dropout         : decimal(6, 6) # dropout probability
    """

    @rowproperty
    def nn(self):
        from fnn.model.recurrents import Rvt

        return Rvt(**self.fetch1())


# -- Recurrent --


@schema.link
class Recurrent:
    links = [Rvt]
    name = "recurrent"


# ----------------------------- Core -----------------------------

# -- Core Base --


class _Core:
    """Core"""

    @rowproperty
    def nn(self):
        """
        Returns
        -------
        fnn.model.cores.Core
            core component
        """
        raise NotImplementedError()


# -- Core Types --


@schema.lookup
class FeedforwardRecurrent(_Core):
    definition = """
    -> Feedforward
    -> Recurrent
    """

    @rowproperty
    def nn(self):
        from fnn.model.cores import FeedforwardRecurrent

        return FeedforwardRecurrent(
            feedforward=(Feedforward & self).link.nn,
            recurrent=(Recurrent() & self).link.nn,
        )


# -- Core --


@schema.link
class Core:
    links = [FeedforwardRecurrent]
    name = "core"
