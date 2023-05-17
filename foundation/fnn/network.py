from djutils import rowproperty
from foundation.fnn.architecture import Architecture, Streams
from foundation.fnn.dataset import VisualSet
from foundation.fnn.dataspec import VisualSpec
from foundation.schemas import fnn as schema


# -------------- Neural Network --------------

# -- Neural Network Base --


class _Network:
    """Neural Network"""

    pass


# -- Neural Network Types --


@schema.lookup
class VisualNetwork(_Network):
    definition = """
    -> VisualSet
    -> VisualSpec
    -> Architecture
    -> Streams
    """


# -- Neural Network Types --


@schema.link
class Network:
    links = [VisualNetwork]
    name = "nn"
    comment = "neural network"


@schema.linkset
class NetworkSet:
    link = Network
    name = "nnset"
    comment = "neural network set"
