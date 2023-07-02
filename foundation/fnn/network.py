from djutils import rowproperty, unique
from foundation.fnn.core import Core
from foundation.fnn.perspective import Perspective
from foundation.fnn.modulation import Modulation
from foundation.fnn.readout import Readout, Unit
from foundation.fnn.shared import Reduce
from foundation.fnn.data import Data
from foundation.schemas import fnn as schema


# ----------------------------- Network -----------------------------

# -- Network Interface --


class NetworkType:
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
    def data_id(self):
        """
        Returns
        -------
        str
            key (foundation.fnn.data.Data)
        """
        raise NotImplementedError()


# -- Network Types --


@schema.lookup
class VisualNetwork(NetworkType):
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
    def data_id(self):
        return self.fetch1("data_id")

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

        data = (Data & self).link.compute
        module._init(
            stimuli=data.stimuli,
            perspectives=data.perspectives,
            modulations=data.modulations,
            units=data.units,
            streams=self.fetch1("streams"),
        )

        return module


# -- Network --


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


# -- Computed Network --


@schema.computed
class NetworkUnit:
    definition = """
    -> Network
    unit_index      : int unsigned  # network unit index
    """

    def make(self, key):
        # network data
        data_id = (Network & key).link.data_id

        # number of units
        units = (Data & {"data_id": data_id}).link.compute.units

        # index keys
        keys = [dict(key, unit_index=i) for i in range(units)]

        # insert
        self.insert(keys)


@schema.computed
class NetworkSetCore:
    definition = """
    -> NetworkSet
    ---
    -> Core
    streams         : int unsigned  # network streams
    network_type    : varchar(128)  # network type
    """

    def make(self, key):
        # network set
        networks = Network & (NetworkSet & key).members
        network_type = unique(networks, "network_type")

        # network set core_id and streams
        keys = getattr(Network, network_type) & networks
        core_id = unique(keys, "core_id")
        streams = unique(keys, "streams")

        # insert
        self.insert1(dict(key, core_id=core_id, streams=streams, network_type=network_type))
