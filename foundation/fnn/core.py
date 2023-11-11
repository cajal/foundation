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
    in_spatial      : int unsigned  # input spatial kernel size
    in_stride       : int unsigned  # input spatial stride
    out_channels    : int unsigned  # output channels per stream
    block_channels  : varchar(128)  # block channels per stream (csv)
    block_groups    : varchar(128)  # block groups per stream (csv)
    block_layers    : varchar(128)  # block layers (csv)
    block_temporals : varchar(128)  # block temporal kernel sizes (csv)
    block_spatials  : varchar(128)  # block spatial kernel sizes (csv)
    block_pools     : varchar(128)  # block spatial pooling sizes (csv)
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


@schema.lookup
class InputDense(FeedforwardType):
    definition = """
    input_spatial   : int unsigned  # input spatial kernel size
    input_stride    : int unsigned  # input spatial stride
    block_channels  : varchar(128)  # block channels per stream (csv)
    block_groups    : varchar(128)  # block groups per stream (csv)
    block_layers    : varchar(128)  # block layers (csv)
    block_temporals : varchar(128)  # block temporal kernel sizes (csv)
    block_spatials  : varchar(128)  # block spatial kernel sizes (csv)
    block_pools     : varchar(128)  # block spatial pooling sizes (csv)
    out_channels    : int unsigned  # output channels per stream
    nonlinear       : varchar(128)  # nonlinearity
    dropout         : decimal(6, 6) # dropout probability
    """

    @rowproperty
    def nn(self):
        from fnn.model.feedforwards import InputDense

        kwargs = self.fetch1("KEY")
        for k, v in kwargs.items():
            if k.startswith("block_"):
                kwargs[k] = v.split(",")

        return InputDense(**kwargs)


# -- Feedforward --


@schema.link
class Feedforward:
    links = [Dense, InputDense]
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
    in_channels         : int unsigned  # in channels per stream
    out_channels        : int unsigned  # out channels per stream
    hidden_channels     : int unsigned  # hidden channels per stream
    common_channels     : int unsigned  # common channels per stream
    groups              : int unsigned  # groups per stream
    spatial             : int unsigned  # spatial kernel size
    init_gate           : decimal(6, 4) # initial gate bias
    dropout             : decimal(6, 6) # dropout probability
    """

    @rowproperty
    def nn(self):
        from fnn.model.recurrents import Rvt

        return Rvt(**self.fetch1())


@schema.lookup
class CvtLstm(RecurrentType):
    definition = """
    in_channels         : int unsigned  # in channels per stream
    out_channels        : int unsigned  # out channels per stream
    hidden_channels     : int unsigned  # hidden channels per stream
    common_channels     : int unsigned  # common channels per stream
    groups              : int unsigned  # groups per stream
    spatial             : int unsigned  # spatial kernel size
    init_input          : decimal(6, 4) # initial input gate bias
    init_forget         : decimal(6, 4) # initial forget gate bias
    dropout             : decimal(6, 6) # dropout probability
    """

    @rowproperty
    def nn(self):
        from fnn.model.recurrents import CvtLstm

        return CvtLstm(**self.fetch1())


@schema.lookup
class ConvLstm(RecurrentType):
    definition = """
    in_channels         : int unsigned  # in channels per stream
    out_channels        : int unsigned  # out channels per stream
    hidden_channels     : int unsigned  # hidden channels per stream
    groups              : int unsigned  # groups per stream
    spatial             : int unsigned  # spatial kernel size
    init_input          : decimal(6, 4) # initial input gate bias
    init_forget         : decimal(6, 4) # initial forget gate bias
    dropout             : decimal(6, 6) # dropout probability
    """

    @rowproperty
    def nn(self):
        from fnn.model.recurrents import ConvLstm

        return ConvLstm(**self.fetch1())


# -- Recurrent --


@schema.link
class Recurrent:
    links = [Rvt, CvtLstm, ConvLstm]
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
