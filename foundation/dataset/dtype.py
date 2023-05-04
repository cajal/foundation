from djutils import row_property
from foundation.utility.stat import SummaryLink
from foundation.utility.standardize import StandardizeLink
from foundation.recording.trace import TraceFilterSet
from foundation.schemas.pipeline import pipe_shared
from foundation.schemas import dataset as schema


# -------------- Data --------------


# -- Data Types --


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


# -- Data Link --


@schema.link
class DataLink:
    links = [ScanUnit, ScanPerspective, ScanModulation]
    name = "data"
    comment = "data type"
