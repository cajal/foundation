from djutils import row_property
from foundation.recording.scan import (
    FilteredScanTrials,
    FilteredScanPerspectives,
    FilteredScanModulations,
    FilteredScanUnits,
)
from foundation.recording.trial import TrialSet
from foundation.recording.trace import TraceSet
from foundation.dataset.dtype import DtypeLink
from foundation.schemas import dataset as schema


# -------------- Recording --------------

# -- Recording Base --


class _Recording:
    """Recording Dataset"""

    @row_property
    def trial_set(self):
        """
        Returns
        -------
        foundation.recording.trial.TrialSet
            single tuple
        """
        raise NotImplementedError()

    @row_property
    def trace_sets(self):
        """
        Yields
        ------
        foundation.dataset.dtype.DtypeLink
            single tuple
        foundation.recording.trace.TraceSet
            single tuple
        """
        raise NotImplementedError()


# -- Recording Types --


@schema.lookup
class Scan(_Recording):
    definition = """
    -> FilteredScanTrials
    -> FilteredScanPerspectives
    -> FilteredScanModulations
    -> FilteredScanUnits
    """

    @row_property
    def trial_set(self):
        return TrialSet & (FilteredScanTrials & self)

    @row_property
    def trace_sets(self):
        for dtype_part, scan_set in [
            [DtypeLink.ScanPerspective, FilteredScanPerspectives],
            [DtypeLink.ScanModulation, FilteredScanModulations],
            [DtypeLink.ScanUnit, FilteredScanUnits],
        ]:
            key = scan_set & self
            dtype = DtypeLink & (dtype_part & key)
            trace_set = TraceSet & key
            yield dtype, trace_set


# -- Recording --


@schema.link
class RecordingLink:
    links = [Scan]
    name = "recording"
    comment = "recording"


# -- Recording Set --


@schema.link_set
class RecordingSet:
    link = RecordingLink
    name = "recordings"
    comment = "recording set"


# -- Computed Recording --


@schema.computed
class RecordingTrials:
    definition = """
    -> RecordingLink
    ---
    -> TrialSet
    """

    def make(self, key):
        trials = (RecordingLink & key).link.trial_set

        trials_id = trials.fetch1("trials_id")

        self.insert1(dict(key, trials_id=trials_id))


@schema.computed
class RecordingTraces:
    definition = """
    -> RecordingLink
    -> DtypeLink
    ---
    -> TraceSet
    """

    @property
    def key_source(self):
        return RecordingLink.proj()

    def make(self, key):
        for dtype, traces in (RecordingLink & key).link.trace_sets:

            dtype_id = dtype.fetch1("dtype_id")
            traces_id = traces.fetch1("traces_id")

            self.insert1(dict(key, dtype_id=dtype_id, traces_id=traces_id))
