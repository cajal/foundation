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
    def data(self):
        """
        Returns
        -------
        foundation.fnn.compute_data.NetworkData (row)
            network data
        """
        raise NotImplementedError()

    @rowproperty
    def module(self):
        """
        Returns
        -------
        fnn.networks.Network
            network module
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
    def data(self):
        return (Data & self).link.data

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
        stimuli, perspectives, modulations, units = self.data.sizes
        module._init(
            stimuli=stimuli,
            perspectives=perspectives,
            modulations=modulations,
            units=units,
            streams=self.fetch1("streams"),
        )
        return module


# -- Neural Network --


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
