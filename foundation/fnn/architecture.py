from djutils import rowproperty
from foundation.fnn.core import Core
from foundation.fnn.readout import Readout, Reduce, Unit
from foundation.fnn.modulation import Modulation
from foundation.fnn.perspective import Perspective
from foundation.schemas import fnn as schema


# -------------- Architecture --------------

# -- Architecture Base --


class _Architecture:
    """Architecture"""

    @rowproperty
    def nn(self):
        """
        Returns
        -------
        fnn.model.architectures.Architecture
            fnn architecture
        """
        raise NotImplementedError()


# -- Architecture Types --


@schema.lookup
class VisualCortex(_Architecture):
    definition = """
    -> Core
    -> Perspective
    -> Modulation
    -> Readout
    -> Reduce
    -> Unit
    """

    @rowproperty
    def nn(self):
        from fnn.model.architectures import VisualCortex

        return VisualCortex(
            core=(Core & self).link.nn,
            perspective=(Perspective & self).link.nn,
            modulation=(Modulation & self).link.nn,
            readout=(Readout & self).link.nn,
            reduce=(Reduce & self).link.nn,
            unit=(Unit & self).link.nn,
        )


# -- Cortex --


@schema.link
class Architecture:
    links = [VisualCortex]
    name = "architecture"
