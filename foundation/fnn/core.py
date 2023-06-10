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
class SpatialTemporalResidual(_Feedforward):
    definition = """
    channels        : varchar(128)  # layer channels (csv)
    spatial_sizes   : varchar(128)  # layer spatial sizes (csv)
    spatial_strides : varchar(128)  # layer spatial strides (csv)
    temporal_sizes  : varchar(128)  # layer temporal sizes (csv)
    nonlinear       : varchar(128)  # nonlinearity
    """

    @rowproperty
    def nn(self):
        from fnn.model.feedforwards import SpatialTemporalResidual

        channels, spatial_sizes, spatial_strides, temporal_sizes, nonlinear = self.fetch1(
            "channels", "spatial_sizes", "spatial_strides", "temporal_sizes", "nonlinear"
        )
        return SpatialTemporalResidual(
            channels=channels.split(","),
            spatial_sizes=spatial_sizes.split(","),
            spatial_strides=spatial_strides.split(","),
            temporal_sizes=temporal_sizes.split(","),
            nonlinear=nonlinear,
        )


# -- Feedforward --


@schema.link
class Feedforward:
    links = [SpatialTemporalResidual]
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

        return FeedforwardRecurrent(
            feedforward=(Feedforward & self).link.nn,
            recurrent=(Recurrent() & self).link.nn,
            channels=self.fetch1("channels"),
        )


# -- Core --


@schema.link
class Core:
    links = [FeedforwardRecurrent]
    name = "core"
