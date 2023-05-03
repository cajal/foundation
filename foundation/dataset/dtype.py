from djutils import row_property
from foundation.scan import unit as scan_unit
from foundation.recording import trial, trace
from foundation.schemas import dataset as schema


# -------------- Data --------------

# -- Data Base --


class _Data:
    """Data Specification"""

    @row_property
    def standardize_link(self):
        """
        Returns
        -------
        foundation.utility.standardize.StandardizeLink
            tuple
        """
        raise NotImplementedError()


# -- Data Types --


@schema.lookup
class ScanUnit:
    definition = """
    -> scan_unit.TrackingMethod
    -> trace.TraceFilterSet
    """
