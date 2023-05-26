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
class Res3d(_Feedforward):
    definition = """
    channels        : varchar(128)  # layer channels (csv)
    kernel_sizes    : varchar(128)  # layer kernel sizes (csv)
    strides         : varchar(128)  # layer strides (csv)
    nonlinear       : varchar(128)  # nonlinearity
    """

    @rowproperty
    def nn(self):
        from fnn.model.feedforwards import Res3d

        c, k, s, n = self.fetch1("channels", "kernel_sizes", "strides", "nonlinear")
        c, k, s = (list(map(int, _.split(","))) for _ in [c, k, s])

        return Res3d(channels=c, kernel_sizes=k, strides=s, nonlinear=n)


# -- Feedforward --


@schema.link
class Feedforward:
    links = [Res3d]
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
    channels        : int unsigned  # channels per stream
    """

    @rowproperty
    def nn(self):
        from fnn.model.cores import FeedforwardRecurrent

        f = (Feedforward & self).link.nn
        r = (Recurrent() & self).link.nn
        c = self.fetch1("channels")

        return FeedforwardRecurrent(feedforward=f, recurrent=r, channels=c)


# -- Core --


@schema.link
class Core:
    links = [FeedforwardRecurrent]
    name = "core"
