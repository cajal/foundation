import numpy as np
import datajoint as dj
from djutils import link, method, row_method
from tqdm import tqdm
from foundation.utils.logging import logger
from foundation.utils.errors import OutOfBounds
from foundation.recordings import trial, trace, resample


schema = dj.schema("foundation_recordings")


# -------------- Nans --------------

# -- Nans Base --


class NansBase:
    """Trace Nans"""

    @row_method
    def nans(self, times, trace):
        """
        Returns
        -------
        foundation.utils.nans.Nans
            callable, detects Nans
        """
        raise NotImplementedError()


# -- Nans Types --


@method(schema)
class ConsecutiveNans(NansBase):
    name = "consecutive_nans"
    comment = "consecutive nans"

    @row_method
    def nans(self, times, trace):
        from foundation.utils.nans import ConsecutiveNans

        return ConsecutiveNans(times, trace)


# -- Nans Link --


@link(schema)
class NansLink:
    links = [ConsecutiveNans]
    name = "nans"
    comment = "trace nans"


# -- Computed Nans --


@schema
class Nans(dj.Computed):
    definition = """
    -> traces.Trace
    -> trials.Trial
    -> resample.Offset
    -> NansLink
    ---
    nans = NULL         : int unsigned  # number of nans
    """

    @property
    def key_source(self):
        return trace.TraceLink.proj() * resample.Offset().proj() * NansLink.proj()
