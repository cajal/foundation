import numpy as np
from djutils import merge
from foundation.utils import tqdm
from foundation.scan.experiment import Scan
from foundation.virtual.bridge import pipe_fuse, pipe_shared, resolve_pipe
from foundation.schemas import scan as schema


@schema.computed
class MeanActivity:
    definition = """
    -> Scan
    -> pipe_fuse.ScanSet.Unit
    -> pipe_shared.SpikeMethod
    ---
    mean_activity = NULL    : float     # mean spiking activity
    """

    @property
    def key_source(self):
        keys = (
            Scan.proj()
            * pipe_shared.PipelineVersion.proj()
            * pipe_shared.SegmentationMethod.proj()
            * pipe_shared.SpikeMethod.proj()
        )
        return keys & pipe_fuse.ScanDone

    def make(self, key):
        # unit activities
        pipe = resolve_pipe(key)
        units = (pipe.Activity.Trace & key).proj("trace")

        # compute acitivity means
        keys = []
        for unit in tqdm(units):
            unit["mean_activity"] = unit.pop("trace").mean()
            keys.append(unit)

        # insert
        self.insert(keys)
