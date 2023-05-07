from djutils import merge, rowproperty
from foundation.virtual import recording
from foundation.dataset.dtype import DtypeLink
from foundation.schemas import dataset as schema


# -------------- Recording --------------

# -- Recording Base --


class _Recording:
    """Recording Dataset"""

    @rowproperty
    def trial_set(self):
        """
        Returns
        -------
        foundation.recording.trial.TrialSet (virtual)
            single tuple
        """
        raise NotImplementedError()

    @rowproperty
    def trace_sets(self):
        """
        Yields
        ------
        foundation.dataset.dtype.DtypeLink  (virtual)
            single tuple
        foundation.recording.trace.TraceSet (virtual)
            single tuple
        """
        raise NotImplementedError()


# -- Recording Types --


@schema.lookup
class Scan(_Recording):
    definition = """
    -> recording.FilteredScanTrials
    -> recording.FilteredScanPerspectives
    -> recording.FilteredScanModulations
    -> recording.FilteredScanUnits
    """

    @rowproperty
    def trial_set(self):
        return recording.TrialSet & (recording.FilteredScanTrials & self)

    @rowproperty
    def trace_sets(self):
        for dtype_part, scan_set in [
            [DtypeLink.ScanPerspective, recording.FilteredScanPerspectives],
            [DtypeLink.ScanModulation, recording.FilteredScanModulations],
            [DtypeLink.ScanUnit, recording.FilteredScanUnits],
        ]:
            key = scan_set & self
            dtype = DtypeLink & merge(key, dtype_part)
            trace_set = recording.TraceSet & key
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
    -> recording.TrialSet
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
    -> recording.TraceSet
    """

    @property
    def key_source(self):
        return RecordingLink.proj()

    def make(self, key):
        for dtype, traces in (RecordingLink & key).link.trace_sets:
            dtype_id = dtype.fetch1("dtype_id")
            traces_id = traces.fetch1("traces_id")
            self.insert1(dict(key, dtype_id=dtype_id, traces_id=traces_id))
