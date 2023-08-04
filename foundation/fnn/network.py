from djutils import rowmethod
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

    @rowmethod
    def module(self, data_id):
        """
        Parameters
        ----------
        str
            key (foundation.fnn.data.Data)

        Returns
        -------
        fnn.networks.Network
            network module
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
    streams     : int unsigned  # network streams
    """

    @rowmethod
    def module(self, data_id):
        from fnn.model.networks import Visual

        module = Visual(
            core=(Core & self).link.nn,
            perspective=(Perspective & self).link.nn,
            modulation=(Modulation & self).link.nn,
            readout=(Readout & self).link.nn,
            reduce=(Reduce & self).link.nn,
            unit=(Unit & self).link.nn,
        )

        data = (Data & {"data_id": data_id}).link.compute
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
    comment = "fnn network"
