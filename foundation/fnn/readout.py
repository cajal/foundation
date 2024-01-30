from djutils import rowproperty
from foundation.fnn.shared import Bound
from foundation.schemas import fnn as schema


# ----------------------------- Position -----------------------------

# -- Position Interface --


class PositionType:
    """Position"""

    @rowproperty
    def nn(self):
        """
        Returns
        -------
        fnn.model.positions.Position
            position component
        """
        raise NotImplementedError()


# -- Position Types --


@schema.lookup
class Gaussian(PositionType):
    definition = """
    init_std            : decimal(6, 4)     # initial stddev
    """

    @rowproperty
    def nn(self):
        from fnn.model.positions import Gaussian

        s = float(self.fetch1("init_std"))
        return Gaussian(init_std=s)


# -- Position --


@schema.link
class Position:
    links = [Gaussian]
    name = "position"


# ----------------------------- Feature -----------------------------

# -- Feature Interface --


class FeatureType:
    """Feature"""

    @rowproperty
    def nn(self):
        """
        Returns
        -------
        fnn.model.features.Feature
            feature component
        """
        raise NotImplementedError()


# -- Feature Types --


@schema.lookup
class Norm(FeatureType):
    definition = """
    groups      : int unsigned  # groups per stream
    """

    @rowproperty
    def nn(self):
        from fnn.model.features import Norm

        return Norm(**self.fetch1())


@schema.method
class Vanilla(FeatureType):
    name = "vanilla"
    comment = "vanilla feature"

    @rowproperty
    def nn(self):
        from fnn.model.features import Vanilla

        return Vanilla()


# -- Feature --


@schema.link
class Feature:
    links = [Norm, Vanilla]
    name = "feature"


# ----------------------------- Readout -----------------------------

# -- Readout Interface --


class ReadoutType:
    """Readout"""

    @rowproperty
    def nn(self):
        """
        Returns
        -------
        fnn.model.readouts.Readout
            readout component
        """
        raise NotImplementedError()


# -- Readout Types --


@schema.lookup
class PositionFeature(ReadoutType):
    definition = """
    -> Position
    -> Bound
    -> Feature
    """

    @rowproperty
    def nn(self):
        from fnn.model.readouts import PositionFeature

        p = (Position & self).link.nn
        b = (Bound & self).link.nn
        f = (Feature & self).link.nn

        return PositionFeature(position=p, bound=b, feature=f)


# -- Readout --


@schema.link
class Readout:
    links = [PositionFeature]
    name = "readout"


# ----------------------------- Unit -----------------------------

# -- Unit Interface --


class UnitType:
    """Unit"""

    @rowproperty
    def nn(self):
        """
        Returns
        -------
        fnn.model.units.Unit
            unit model
        """
        raise NotImplementedError()


# -- Unit Types --


@schema.method
class Poisson(UnitType):
    name = "poisson"
    comment = "poisson unit"

    @rowproperty
    def nn(self):
        from fnn.model.units import Poisson

        return Poisson()


# -- Unit --


@schema.link
class Unit:
    links = [Poisson]
    name = "unit"
