from djutils import merge, rowproperty
from foundation.virtual import scan, utility
from foundation.virtual.bridge import pipe_fuse, pipe_shared, pipe_stim, pipe_tread, resolve_pipe
from foundation.recording.trial import Trial, TrialSet, TrialBounds
from foundation.schemas import recording as schema


# -------------- Trace --------------

# -- Trace Base --


class _Trace:
    """Recording Trace"""

    @rowproperty
    def trial_set(self):
        """
        Returns
        -------
        foundation.recording.trial.TrialSet
            tuple
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
    def trial_set(self):
        key = pipe_stim.Trial.proj() & self
        key = merge(key, Trial.ScanTrial)
        key = TrialSet.fill(key, prompt=False, silent=True)
        return TrialSet & key


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
        key["trialset_id"] = (Trace & key).link.trial_set.fetch1("trialset_id")
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
        from foundation.recording.compute import SummarizeTrace

        key["summary"] = (SummarizeTrace & key).statistic
        self.insert1(key)


# -------------- Trace Filter --------------

# -- Filter Types --

# -- Filter --


@schema.filterlink
class TraceFilter:
    links = []
    name = "trace_filter"
    comment = "trace filter"


@schema.filterlinkset
class TraceFilterSet:
    link = TraceFilter
    name = "trace_filterset"
    comment = "trace filter set"
