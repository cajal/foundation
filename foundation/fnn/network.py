from djutils import rowproperty
from foundation.fnn.core import Core
from foundation.fnn.readout import Readout, Reduce, Unit
from foundation.fnn.modulation import Modulation
from foundation.fnn.perspective import Perspective
from foundation.fnn.data import Data
from foundation.utils import logger
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


# -- Computed Neural Network --


@schema.computed
class ParallelNetworkSet:
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
        network_type = networks.fetch("network_type")
        try:
            (network_type,) = set(network_type)
        except ValueError as e:
            logger.warning(e)
            logger.warning(f"Skipping {key}")
            return

        # network set core_id and streams
        keys = getattr(Network, network_type) & networks
        core_id, streams = keys.fetch("core_id", "streams")
        try:
            (core_id,) = set(core_id)
            (streams,) = set(streams)
        except ValueError as e:
            logger.warning(e)
            logger.warning(f"Skipping {key}")
            return

        # insert
        self.insert1(dict(key, core_id=core_id, streams=streams, network_type=network_type))
