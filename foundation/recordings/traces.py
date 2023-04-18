import numpy as np
import datajoint as dj
from djutils import link, group, method, row_method, row_property, MissingError
from foundation.recordings import trials
from foundation.utils.logging import logger

pipe_meso = dj.create_virtual_module("pipe_meso", "pipeline_meso")
schema = dj.schema("foundation_recordings")


# -------------- Trace --------------

# -- Base --


class TraceBase:
    """Recording Trace"""

    @row_property
    def timed_trace(self):
        """
        Returns
        -------
        1D array
            recording trace
        1D array
            recording times of each point in the trace

        IMPORTANT : arrays must be the same length
        """
        raise NotImplementedError()

    @row_property
    def trials(self):
        """
        Returns
        -------
        Trial
            Trial tuples
        """
        raise NotImplementedError()


# -- Types --


@schema
class MesoActivity(TraceBase, dj.Lookup):
    definition = """
    -> pipe_meso.Activity.Trace
    """
