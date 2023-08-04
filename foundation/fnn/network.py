from djutils import rowproperty, rowmethod
from foundation.fnn.core import Core
from foundation.fnn.perspective import Perspective
from foundation.fnn.modulation import Modulation
from foundation.fnn.readout import Readout, Unit
from foundation.fnn.shared import Reduce
from foundation.fnn.data import Data
from foundation.schemas import fnn as schema


# ----------------------------- Module -----------------------------


@schema.lookup
class Module:
    definition = """
    module      : varchar(128)  # module name
    """


@schema.set
class ModuleSet:
    keys = [Module]
    name = "moduleset"
    comment = "module set"


# ----------------------------- Network -----------------------------

# -- Network Interface --


class NetworkType:
    """Neural Network"""

    @rowproperty
    def modules(self):
        """
        Returns
        -------
        List[str]
            list of module names
        """
        raise NotImplementedError()

    @rowmethod
    def network(self, data_id):
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

    @rowproperty
    def modules(self):
        return ["core", "perspective", "modulation", "readout", "reduce", "unit"]

    @rowmethod
    def network(self, data_id):
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


# -- Computed Network --


@schema.computed
class NetworkModule:
    definition = """
    -> Network
    -> Module
    """

    @property
    def key_source(self):
        return Network.proj()

    def make(self, key):
        # modules
        modules = (Network & key).link.modules

        # keys
        keys = [dict(key, module=m) for m in modules]

        # insert module
        Module.insert(keys, skip_duplicates=True, ignore_extra_fields=True)

        # insert network module
        self.insert(keys)
