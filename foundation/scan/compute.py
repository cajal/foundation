import numpy as np
from djutils import keys, merge, rowproperty
from foundation.utils import logger
from foundation.virtual.bridge import pipe_fuse, pipe_shared, resolve_pipe
from foundation.scan.unit import UnitSet, FilteredUnits


@keys
class LoadActivity:
    """Scan unit set activity"""

    @property
    def key_list(self):
        return [
            UnitSet & FilteredUnits,
            pipe_shared.SpikeMethod & "spike_method in (5, 6)",
        ]

    @rowproperty
    def activity(self):
        """
        Returns
        -------
        2D array -- [units, samples]
            scan unit activity, units ordered by 'units_index'
        """
        traces = merge(
            (UnitSet & self.key).members,
            resolve_pipe(self.key).Activity.Trace & self.key,
        )
        logger.info(f"Loading {len(traces)} activity traces")
        traces = traces.fetch("trace", order_by="units_index")
        return np.stack(traces)
