from djutils import rowproperty
from foundation.virtual.bridge import pipe_shared
from foundation.virtual import recording
from foundation.schemas import dataset as schema


# -------------- Dtype --------------


# -- Dtype Types --


@schema.lookup
class ScanUnit:
    definition = """
    -> pipe_shared.SpikeMethod
    -> recording.TraceFilterSet
    """


@schema.lookup
class ScanPerspective:
    definition = """
    -> pipe_shared.TrackingMethod
    -> recording.TraceFilterSet
    """


@schema.lookup
class ScanModulation:
    definition = """
    -> pipe_shared.TrackingMethod
    -> recording.TraceFilterSet
    """


# -- Dtype --


@schema.link
class DtypeLink:
    links = [ScanUnit, ScanPerspective, ScanModulation]
    name = "dtype"
    comment = "data type"
