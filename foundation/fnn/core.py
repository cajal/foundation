from djutils import rowproperty
from foundation.schemas import fnn as schema


# ----------------------------- Feedforward -----------------------------

# -- Feedforward Interface --


class FeedforwardType:
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
class Dense(FeedforwardType):
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

# -- Recurrent Interface --


class RecurrentType:
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
class Rvt(RecurrentType):
    definition = """
    recurrent_channels  : int unsigned  # recurrent channels per stream
    attention_channels  : int unsigned  # attention channels per stream
    out_channels        : int unsigned  # out channels per stream
    groups              : int unsigned  # groups per stream
    heads               : int unsigned  # heads per stream
    kernel_size         : int unsigned  # kernel size
    dropout             : decimal(6, 6) # dropout probability
    """

    @rowproperty
    def nn(self):
        from fnn.model.recurrents import Rvt

        return Rvt(**self.fetch1())


@schema.lookup
class ConvLstm(RecurrentType):
    definition = """
    recurrent_channels  : int unsigned  # recurrent channels per stream
    out_channels        : int unsigned  # out channels per stream
    groups              : int unsigned  # groups per stream
    kernel_size         : int unsigned  # kernel size
    dropout             : decimal(6, 6) # dropout probability
    """

    @rowproperty
    def nn(self):
        from fnn.model.recurrents import ConvLstm

        return ConvLstm(**self.fetch1())


# -- Recurrent --


@schema.link
class Recurrent:
    links = [Rvt, ConvLstm]
    name = "recurrent"


# ----------------------------- Core -----------------------------

# -- Core Interface --


class CoreType:
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
class FeedforwardRecurrent(CoreType):
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
