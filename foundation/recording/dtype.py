from djutils import rowproperty
from foundation.virtual.bridge import pipe_shared
from foundation.virtual import utility
from foundation.recording.trace import TraceFilterSet
from foundation.schemas import recording as schema


# -------------- Dtype --------------

# -- Dtype Base --


class _Dtype:
    """Recording Data Type"""

    @rowproperty
    def loader(self):
        """
        Returns
        -------
        djutils.derived.Keys
            keys with `load` rowproperty
        """


# -- Dtype Types --


@schema.lookup
class VideoStimulus(_Dtype):
    definition = """
    -> utility.Resize
    -> utility.Resolution
    """


@schema.lookup
class ScanPerspectiveTraces(_Dtype):
    definition = """
    -> pipe_shared.TrackingMethod
    -> TraceFilterSet
    -> utility.Standardize
    -> utility.Resample
    -> utility.Offset
    """


@schema.lookup
class ScanModulationTraces(_Dtype):
    definition = """
    -> pipe_shared.TrackingMethod
    -> TraceFilterSet
    -> utility.Standardize
    -> utility.Resample
    -> utility.Offset
    """


@schema.lookup
class ScanUnitTraces(_Dtype):
    definition = """
    -> pipe_shared.SpikeMethod
    -> TraceFilterSet
    -> utility.Standardize
    -> utility.Resample
    -> utility.Offset
    """


# -- Dtype --


@schema.link
class Dtype:
    links = [VideoStimulus, ScanPerspectiveTraces, ScanModulationTraces, ScanUnitTraces]
    name = "dtype"
    comment = "data type"
