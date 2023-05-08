from djutils import merge, rowproperty
from foundation.virtual import recording
from foundation.dataset.dtype import Dtype
from foundation.schemas import dataset as schema


# -------------- Recording --------------

# -- Recording Base --


class _Recording:
    """Recording Dataset"""

    @rowproperty
    def trialset(self):
        """
        Returns
        -------
        foundation.recording.trial.TrialSet (virtual)
            single tuple
        """
        raise NotImplementedError()

    @rowproperty
    def tracesets(self):
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
    -> recording.ScanTrials
    -> recording.ScanPerspectives
    -> recording.ScanModulations
    -> recording.ScanUnits
    """

    @rowproperty
    def trialset(self):
        return recording.TrialSet & (recording.ScanTrials & self)

    @rowproperty
    def tracesets(self):
        for dtype_part, scan_set in [
            [Dtype.ScanPerspective, recording.ScanPerspectives],
            [Dtype.ScanModulation, recording.ScanModulations],
            [Dtype.ScanUnit, recording.ScanUnits],
        ]:
            key = scan_set & self
            dtype = Dtype & merge(key, dtype_part)
            trace_set = recording.TraceSet & key
            yield dtype, trace_set


# -- Recording --


@schema.link
class Recording:
    links = [Scan]
    name = "recording"
    comment = "recording"


# -- Recording Set --


@schema.linkset
class RecordingSet:
    link = Recording
    name = "recordingset"
    comment = "recording set"


# -- Computed Recording --


@schema.computed
class RecordingTrials:
    definition = """
    -> Recording
    ---
    -> recording.TrialSet
    """

    def make(self, key):
        trials = (Recording & key).link.trialset
        trialset_id = trials.fetch1("trialset_id")
        self.insert1(dict(key, trialset_id=trialset_id))


@schema.computed
class RecordingTraces:
    definition = """
    -> Recording
    -> Dtype
    ---
    -> recording.TraceSet
    """

    @property
    def key_source(self):
        return Recording.proj()

    def make(self, key):
        for dtype, traces in (Recording & key).link.tracesets:
            dtype_id = dtype.fetch1("dtype_id")
            traceset_id = traces.fetch1("traceset_id")
            self.insert1(dict(key, dtype_id=dtype_id, traceset_id=traceset_id))
