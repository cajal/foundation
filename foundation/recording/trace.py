from djutils import merge, row_property
from foundation.scan.experiment import Scan
from foundation.scan.pupil import PupilTrace
from foundation.recording.trial import TrialLink, TrialSet, TrialBounds
from foundation.virtual.bridge import pipe_fuse, pipe_shared, pipe_stim, pipe_tread, resolve_pipe
from foundation.schemas import recording as schema


# -------------- Trace --------------

# -- Trace Base --


class _Trace:
    """Recording Trace"""

    @row_property
    def trial_set(self):
        """
        Returns
        -------
        foundation.recording.trial.TrialSet
            tuple
        """
        raise NotImplementedError()

    @row_property
    def times(self):
        """
        Returns
        -------
        1D array
            trace times
        """
        raise NotImplementedError()

    @row_property
    def values(self):
        """
        Returns
        -------
        1D array
            trace values
        """
        raise NotImplementedError()

    @row_property
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

    @row_property
    def trial_set(self):
        key = pipe_stim.Trial.proj() & self
        key = merge(key, TrialLink.ScanTrial)
        key = TrialSet.fill(key, prompt=False, silent=True)
        return TrialSet & key


@schema.lookup
class ScanUnit(_Scan):
    definition = """
    -> Scan
    -> pipe_fuse.ScanSet.Unit
    -> pipe_shared.SpikeMethod
    """

    @row_property
    def times(self):
        times = (Scan & self).fetch1("scan_times")
        delay = (resolve_pipe(self).ScanSet.UnitInfo & self).fetch1("ms_delay") / 1000
        return times + delay

    @row_property
    def values(self):
        return (resolve_pipe(self).Activity.Trace & self).fetch1("trace").clip(0)

    @row_property
    def homogeneous(self):
        return True


@schema.lookup
class ScanPupil(_Scan):
    definition = """
    -> PupilTrace
    """

    @row_property
    def times(self):
        return (Scan & self).fetch1("eye_times")

    @row_property
    def values(self):
        return (PupilTrace & self).fetch1("pupil_trace")

    @row_property
    def homogeneous(self):
        return False


@schema.lookup
class ScanTreadmill(_Scan):
    definition = """
    -> Scan
    -> pipe_tread.Treadmill
    """

    @row_property
    def times(self):
        return (Scan & self).fetch1("treadmill_times")

    @row_property
    def values(self):
        return (pipe_tread.Treadmill & self).fetch1("treadmill_vel")

    @row_property
    def homogeneous(self):
        return True


# -- Trace --


@schema.link
class TraceLink:
    links = [ScanUnit, ScanPupil, ScanTreadmill]
    name = "trace"
    comment = "trace"


# -- Trace Set --


@schema.link_set
class TraceSet:
    link = TraceLink
    name = "traces"
    comment = "trace set"


# -- Computed Trace --


@schema.computed
class TraceHomogeneous:
    definition = """
    -> TraceLink
    ---
    homogeneous     : bool      # homogeneous tranformation
    """

    def make(self, key):
        key["homogeneous"] = (TraceLink & key).link.homogeneous
        self.insert1(key)


@schema.computed
class TraceTrials:
    definition = """
    -> TraceLink
    ---
    -> TrialSet
    """

    def make(self, key):
        key["trials_id"] = (TraceLink & key).link.trial_set.fetch1("trials_id")
        self.insert1(key)


# -------------- Trace Filter --------------

# -- Filter Types --

# -- Filter --


@schema.filter_link
class TraceFilterLink:
    links = []
    name = "trace_filter"
    comment = "trace filter"


# -- Filter Set --


@schema.filter_link_set
class TraceFilterSet:
    link = TraceFilterLink
    name = "trace_filters"
    comment = "trace filter set"
