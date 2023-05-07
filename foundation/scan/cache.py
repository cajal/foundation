import numpy as np
from djutils import Filepath, rowmethod
from foundation.virtual.bridge import pipe_shared
from foundation.scan.unit import UnitSet
from foundation.schemas import scan as schema


@schema.computed
class UnitsActivity(Filepath):
    definition = """
    -> UnitSet
    -> pipe_shared.SpikeMethod
    ---
    activity        : filepath@scratch09    # npy file, [units, samples]
    """

    def make(self, key):
        from foundation.scan.compute import LoadActivity

        # load activity
        activity = (LoadActivity & key).activity

        # save activity
        filepath = self.createpath(key, "activity", "npy")
        np.save(filepath, activity)

        # insert key
        self.insert1(dict(key, activity=filepath))

    @rowmethod
    def load(self, checksum=True):
        filepath = self.filepath("activity", checksum=checksum)
        return np.load(filepath, mmap_mode="r")
