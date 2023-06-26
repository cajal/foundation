from djutils import merge, rowproperty, rowmethod
from foundation.virtual import utility, scan
from foundation.virtual.bridge import pipe_fuse, pipe_shared, pipe_stim, pipe_tread, resolve_pipe
from foundation.recording.trial import Trial, TrialSet, TrialBounds
from foundation.schemas import recording as schema


# ---------------------------- Trace ----------------------------

# -- Trace Base --


class TraceType:
    """Recording Trace"""

    @rowproperty
    def compute(self):
        """
        Returns
        -------
        foundation.recording.compute_trace.TraceType (row)
            compute trace
        """
        raise NotImplementedError()


# -- Trace Types --


@schema.lookup
class ScanUnit(TraceType):
    definition = """
    -> scan.Scan
    -> pipe_fuse.ScanSet.Unit
    -> pipe_shared.SpikeMethod
    """

    @rowproperty
    def compute(self):
        from foundation.recording.compute_trace import ScanUnit

        return ScanUnit & self


@schema.lookup
class ScanPupil(TraceType):
    definition = """
    -> scan.PupilTrace
    """

    @rowproperty
    def compute(self):
        from foundation.recording.compute_trace import ScanPupil

        return ScanPupil & self


@schema.lookup
class ScanTreadmill(TraceType):
    definition = """
    -> scan.Scan
    -> pipe_tread.Treadmill
    """

    @rowproperty
    def compute(self):
        from foundation.recording.compute_trace import ScanTreadmill

        return ScanTreadmill & self


# -- Trace --


@schema.link
class Trace:
    links = [ScanUnit, ScanPupil, ScanTreadmill]
    name = "trace"
    comment = "recording trace"


@schema.linkset
class TraceSet:
    link = Trace
    name = "traceset"
    comment = "trace set"


# -- Computed Trace --


@schema.computed
class TraceTrials:
    definition = """
    -> Trace
    ---
    -> TrialSet
    """

    def make(self, key):
        key["trialset_id"] = (Trace & key).link.compute.trialset_id
        self.insert1(key)


@schema.computed
class TraceHomogeneous:
    definition = """
    -> Trace
    ---
    homogeneous     : bool      # homogeneous | unrestricted transform
    """

    def make(self, key):
        key["homogeneous"] = (Trace & key).link.compute.homogeneous
        self.insert1(key)


@schema.computed
class TraceSummary:
    definition = """
    -> Trace
    -> TrialSet
    -> utility.Summary
    -> utility.Resample
    -> utility.Offset
    -> utility.Rate
    ---
    summary = NULL      : float     # summary statistic
    """

    def make(self, key):
        from foundation.recording.compute_trace import TraceSummary

        key["summary"] = (TraceSummary & key).statistic
        self.insert1(key)


# ---------------------------- Trace Filter ----------------------------

# -- Trace Filter Types --


@schema.lookupfilter
class ScanUnitFilter:
    filtertype = Trace
    definition = """
    -> pipe_shared.ClassificationMethod
    -> pipe_shared.MaskType
    """

    @rowmethod
    def filter(self, traces):
        units = merge(traces, Trace.ScanUnit)
        pipe = resolve_pipe(units)
        key = units * pipe.ScanSet.Unit * pipe.MaskClassification.Type & self.fetch1()

        return traces & key.proj()


# -- Trace Filter --


@schema.filterlink
class TraceFilter:
    links = [ScanUnitFilter]
    name = "trace_filter"
    comment = "trace filter"


@schema.filterlinkset
class TraceFilterSet:
    link = TraceFilter
    name = "trace_filterset"
    comment = "trace filter set"
