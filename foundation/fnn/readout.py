from djutils import rowproperty
from foundation.schemas import fnn as schema


# -------------- Position --------------

# -- Position Base --


class _Position:
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
class Gaussian(_Position):
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


# -------------- Bound --------------

# -- Bound Base --


class _Bound:
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
class Tanh(_Bound):
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


# -------------- Feature --------------

# -- Feature Base --


class _Feature:
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


@schema.method
class Norm(_Feature):
    name = "norm"
    comment = "normalized feature"

    @rowproperty
    def nn(self):
        from fnn.model.features import Norm

        return Norm()


# -- Feature --


@schema.link
class Feature:
    links = [Norm]
    name = "feature"


# -------------- Readout --------------

# -- Readout Base --


class _Readout:
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
class PositionFeature(_Readout):
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
