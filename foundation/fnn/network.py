from djutils import rowproperty
from foundation.fnn.architecture import Architecture, Streams
from foundation.fnn.dataset import VisualSet
from foundation.fnn.dataspec import VisualSpec
from foundation.schemas import fnn as schema


# -------------- Neural Network --------------

# -- Neural Network Base --


class _Network:
    """Neural Network"""

    @rowproperty
    def dataset(self):
        """
        Returns
        -------
        fnn.data.datset.Dataset
            network training dataset
        """
        raise NotImplementedError()

    @rowproperty
    def module(self):
        """
        Returns
        -------
        fnn.modules.Module
            network module
        """
        raise NotImplementedError()


# -- Neural Network Types --


@schema.lookup
class VisualNetwork(_Network):
    definition = """
    -> VisualSet
    -> VisualSpec
    -> Architecture
    -> Streams
    """

    @rowproperty
    def datakey(self):
        keys = (VisualSet & self).link.data_keys
        keys &= (VisualSpec & self).link.data_keys
        keys &= (Architecture & self).link.data_keys

        (keys,) = keys
        return keys & self

    @rowproperty
    def dataset(self):
        return self.datakey.dataset

    @rowproperty
    def module(self):
        sizes = self.datakey.sizes
        streams = self.fetch1("streams")

        nn = (Architecture & self).link.nn
        nn._init(**sizes, streams=streams)

        return nn


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
