from djutils import rowproperty
from foundation.fnn.core import Core
from foundation.fnn.readout import Readout, Reduce, Unit
from foundation.fnn.modulation import Modulation
from foundation.fnn.perspective import Perspective
from foundation.fnn.data import Data
from foundation.schemas import fnn as schema


# ----------------------------- Neural Network -----------------------------

# -- Neural Network Base --


class _Network:
    """Neural Network"""

    @rowproperty
    def module(self):
        """
        Returns
        -------
        fnn.networks.Network
            network module
        """
        raise NotImplementedError()

    @rowproperty
    def data(self):
        """
        Returns
        -------
        foundation.fnn.data.Data
            tuple, network data
        """
        raise NotImplementedError()


# -- Neural Network Types --


@schema.lookup
class VisualNetwork(_Network):
    definition = """
    -> Core
    -> Perspective
    -> Modulation
    -> Readout
    -> Reduce
    -> Unit
    -> Data
    streams     : int unsigned  # network streams
    """

    @rowproperty
    def module(self):
        from fnn.model.networks import Visual

        module = Visual(
            core=(Core & self).link.nn,
            perspective=(Perspective & self).link.nn,
            modulation=(Modulation & self).link.nn,
            readout=(Readout & self).link.nn,
            reduce=(Reduce & self).link.nn,
            unit=(Unit & self).link.nn,
        )
        module._init(
            **(Data & self).link.sizes,
            streams=self.fetch1("streams"),
        )
        return module

    @rowproperty
    def data(self):
        return Data & self


# -- Neural Network Types --


@schema.link
class Network:
    links = [VisualNetwork]
    name = "network"
    comment = "neural network"


@schema.linkset
class NetworkSet:
    link = Network
    name = "networkset"
    comment = "neural network set"
