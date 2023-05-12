from djutils import rowproperty
from foundation.fnn.core import Core
from foundation.fnn.readout import Readout
from foundation.fnn.modulation import Modulation
from foundation.fnn.perspective import Perspective
from foundation.schemas import fnn as schema


# -------------- Unit --------------

# -- Unit Base --


class _Unit:
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
class Poisson(_Unit):
    name = "poission"
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


# -------------- Cortex --------------

# -- Cortex Base --


class _Cortex:
    """Cortex"""

    @rowproperty
    def nn(self):
        """
        Returns
        -------
        fnn.model.cortex.Cortex
            cortex model
        """
        raise NotImplementedError()


# -- Cortex Types --


@schema.lookup
class Visual(_Cortex):
    definition = """
    -> Unit
    -> Core
    -> Readout
    -> Modulation
    -> Perspective
    streams         : int unsigned  # number of streams
    """


# -- Cortex --


@schema.link
class Cortex:
    links = [Visual]
    name = "cortex"
