from djutils import rowproperty
from foundation.schemas import fnn as schema


# ----------------------------- Position -----------------------------

# -- Position Base --


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


# ----------------------------- Bound -----------------------------

# -- Bound Base --


class BoundType:
    """Bound"""

    @rowproperty
    def nn(self):
        """
        Returns
        -------
        fnn.model.bounds.Bound
            bound component
        """
        raise NotImplementedError()


# -- Bound Types --


@schema.method
class Tanh(BoundType):
    name = "tanh"
    comment = "tanh bound"

    @rowproperty
    def nn(self):
        from fnn.model.bounds import Tanh

        return Tanh()


# -- Bound --


@schema.link
class Bound:
    links = [Tanh]
    name = "bound"


# ----------------------------- Feature -----------------------------

# -- Feature Base --


class FeatureType:
    """Feature"""

    @rowproperty
    def nn(self):
        """
        Returns
        -------
        fnn.model.bounds.Bound
            bound component
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


# -- Feature --


@schema.link
class Feature:
    links = [Norm]
    name = "feature"


# ----------------------------- Readout -----------------------------

# -- Readout Base --


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

# -- Unit Base --


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


# ----------------------------- Reduce -----------------------------

# -- Reduce Base --


class ReduceType:
    """Reduce"""

    @rowproperty
    def nn(self):
        """
        Returns
        -------
        fnn.model.reductions.Reduce
            reduction method
        """
        raise NotImplementedError()


# -- Unit Types --


@schema.method
class Mean(ReduceType):
    name = "mean"
    comment = "mean reduction"

    @rowproperty
    def nn(self):
        from fnn.model.reductions import Mean

        return Mean()


# -- Unit --


@schema.link
class Reduce:
    links = [Mean]
    name = "reduce"
