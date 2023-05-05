from djutils import row_property
from foundation.utility.stat import SummaryLink
from foundation.recording.trace import TraceFilterSet
from foundation.schemas.pipeline import pipe_shared
from foundation.schemas import dataset as schema


# -------------- Dtype --------------


# -- Dtype Types --


@schema.lookup
class ScanUnit:
    definition = """
    -> pipe_shared.SpikeMethod
    -> TraceFilterSet
    """


@schema.lookup
class ScanPerspective:
    definition = """
    -> pipe_shared.TrackingMethod
    -> TraceFilterSet
    """


@schema.lookup
class ScanModulation:
    definition = """
    -> pipe_shared.TrackingMethod
    -> TraceFilterSet
    """


# -- Dtype --


@schema.link
class DtypeLink:
    links = [ScanUnit, ScanPerspective, ScanModulation]
    name = "dtype"
    comment = "data type"
