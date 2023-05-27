from djutils import merge, rowproperty, rowmethod
from foundation.virtual import utility, scan
from foundation.virtual.bridge import pipe_fuse, pipe_shared, pipe_stim, pipe_tread, resolve_pipe
from foundation.recording.trial import Trial, TrialSet, TrialBounds
from foundation.schemas import recording as schema


# ---------------------------- Trace ----------------------------

# -- Trace Base --


class _Trace:
    """Recording Trace"""

    @rowproperty
    def trialset_id(self):
        """
        Returns
        -------
        str
            trialset_id (foundation.recording.trial.TrialSet)
        """
        raise NotImplementedError()

    @rowproperty
    def times(self):
        """
        Returns
        -------
        1D array
            trace times
        """
        raise NotImplementedError()

    @rowproperty
    def values(self):
        """
        Returns
        -------
        1D array
            trace values
        """
        raise NotImplementedError()

    @rowproperty
    def homogeneous(self):
        """
        Returns
        -------
        bool
            homogeneous transformation
        """
        raise NotImplementedError()


# -- Trace Types --


class _Scan(_Trace):
    """Scan Trace"""

    @rowproperty
    def trialset_id(self):
        key = pipe_stim.Trial.proj() & self
        key = merge(key, Trial.ScanTrial)
        key = TrialSet.fill(key, prompt=False, silent=True)
        return key["trialset_id"]


@schema.lookup
class ScanUnit(_Scan):
    definition = """
    -> scan.Scan
    -> pipe_fuse.ScanSet.Unit
    -> pipe_shared.SpikeMethod
    """

    @rowproperty
    def times(self):
        times = (scan.Scan & self).fetch1("scan_times")
        delay = (resolve_pipe(self).ScanSet.UnitInfo & self).fetch1("ms_delay") / 1000
        return times + delay

    @rowproperty
    def values(self):
        return (resolve_pipe(self).Activity.Trace & self).fetch1("trace").clip(0)

    @rowproperty
    def homogeneous(self):
        return True


@schema.lookup
class ScanPupil(_Scan):
    definition = """
    -> scan.PupilTrace
    """

    @rowproperty
    def times(self):
        return (scan.Scan & self).fetch1("eye_times")

    @rowproperty
    def values(self):
        return (scan.PupilTrace & self).fetch1("pupil_trace")

    @rowproperty
    def homogeneous(self):
        return False


@schema.lookup
class ScanTreadmill(_Scan):
    definition = """
    -> scan.Scan
    -> pipe_tread.Treadmill
    """

    @rowproperty
    def times(self):
        return (scan.Scan & self).fetch1("treadmill_times")

    @rowproperty
    def values(self):
        return (pipe_tread.Treadmill & self).fetch1("treadmill_vel")

    @rowproperty
    def homogeneous(self):
        return True


# -- Trace --


@schema.link
class Trace:
    links = [ScanUnit, ScanPupil, ScanTreadmill]
    name = "trace"
    comment = "trace"


@schema.linkset
class TraceSet:
    link = Trace
    name = "traceset"
    comment = "trace set"


# -- Computed Trace --


@schema.computed
class TraceHomogeneous:
    definition = """
    -> Trace
    ---
    homogeneous     : bool      # homogeneous tranformation
    """

    def make(self, key):
        key["homogeneous"] = (Trace & key).link.homogeneous
        self.insert1(key)


@schema.computed
class TraceTrials:
    definition = """
    -> Trace
    ---
    -> TrialSet
    """

    def make(self, key):
        key["trialset_id"] = (Trace & key).link.trialset_id
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

# -- Trace Filter Base --


class _TraceFilter:
    filtertype = Trace


# -- Trace Filter Types --


@schema.lookupfilter
class ScanUnitFilter(_TraceFilter):
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
