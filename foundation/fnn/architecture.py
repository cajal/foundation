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

    @rowproperty
    def datakeys(self):
        """
        Returns
        -------
        set[djutils.derived.Keys]
            keys with `dataset` rowproperty
        """
        raise NotImplementedError()


# -- Architecture Types --


@schema.lookup
class VisualArchitecture(_Architecture):
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
        from fnn.model.architectures import Visual

        return Visual(
            core=(Core & self).link.nn,
            perspective=(Perspective & self).link.nn,
            modulation=(Modulation & self).link.nn,
            readout=(Readout & self).link.nn,
            reduce=(Reduce & self).link.nn,
            unit=(Unit & self).link.nn,
        )

    @rowproperty
    def datakeys(self):
        from foundation.fnn.compute import ResampledVisualRecording

        return {ResampledVisualRecording}


# -- Architecture --


@schema.link
class Architecture:
    links = [VisualArchitecture]
    name = "architecture"


@schema.lookup
class Streams:
    definition = """
    streams     : int unsigned  # architecture streams
    """
