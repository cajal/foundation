from djutils import rowproperty
from foundation.fnn.core import Core
from foundation.fnn.readout import Readout, Reduce, Unit
from foundation.fnn.modulation import Modulation
from foundation.fnn.perspective import Perspective
from foundation.fnn.data import Data
from foundation.schemas import fnn as schema


# -------------- Neural Network --------------

# -- Neural Network Base --


class _Network:
    """Neural Network"""

    @rowproperty
    def module(self):
        """
        Returns
        -------
        fnn.modules.Module
            network module
        """
        raise NotImplementedError()

    @rowproperty
    def dataset(self):
        """
        Returns
        -------
        fnn.data.datset.Dataset
            training dataset
        """
        raise NotImplementedError()

    @rowproperty
    def response_timing(self):
        """
        Returns
        -------
        float
            sampling period (seconds)
        float
            sampling offset (seconds)
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

        streams = self.fetch1("streams")
        sizes = (Data & self).link.network_sizes
        module = Visual(
            core=(Core & self).link.nn,
            perspective=(Perspective & self).link.nn,
            modulation=(Modulation & self).link.nn,
            readout=(Readout & self).link.nn,
            reduce=(Reduce & self).link.nn,
            unit=(Unit & self).link.nn,
        )
        module._init(**sizes, streams=streams)
        return module

    @rowproperty
    def dataset(self):
        return (Data & self).link.dataset


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
