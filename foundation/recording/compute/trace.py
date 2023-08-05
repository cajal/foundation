import numpy as np
from djutils import keys, merge, rowmethod, rowproperty
from foundation.virtual.bridge import pipe_fuse, pipe_shared, pipe_tread, resolve_pipe
from foundation.virtual import scan, recording


# ----------------------------- Trace -----------------------------

# -- Trace Interface --


class TraceType:
    """Recording Trace"""

    @rowproperty
    def trialset_id(self):
        """
        Returns
        -------
        str
            key (foundation.recording.trial.TrialSet)
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
            homogeneous | unrestricted transform
        """
        raise NotImplementedError()


class ScanTraceType(TraceType):
    """Scan Trace"""

    @rowproperty
    def trialset_id(self):
        return merge(self.key, recording.ScanRecording).fetch1("trialset_id")


# -- Trace Types --


@keys
class ScanUnit(ScanTraceType):
    """Scan Unit Trace"""

    @property
    def keys(self):
        return [
            scan.Scan,
            pipe_fuse.ScanSet.Unit,
            pipe_shared.SpikeMethod,
        ]

    @rowproperty
    def times(self):
        times = (scan.Scan & self.item).fetch1("scan_times")
        delay = (resolve_pipe(self.item).ScanSet.UnitInfo & self.item).fetch1("ms_delay") / 1000
        return times + delay

    @rowproperty
    def values(self):
        return (resolve_pipe(self.item).Activity.Trace & self.item).fetch1("trace").clip(0)

    @rowproperty
    def homogeneous(self):
        return True


@keys
class ScanPupil(ScanTraceType):
    """Scan Pupil Trace"""

    @property
    def keys(self):
        return [
            scan.PupilTrace,
        ]

    @rowproperty
    def times(self):
        return (scan.Scan & self.item).fetch1("eye_times")

    @rowproperty
    def values(self):
        return (scan.PupilTrace & self.item).fetch1("pupil_trace")

    @rowproperty
    def homogeneous(self):
        return False


@keys
class ScanTreadmill(ScanTraceType):
    """Scan Treadmill Trace"""

    @property
    def keys(self):
        return [
            scan.Scan,
            pipe_tread.Treadmill,
        ]

    @rowproperty
    def times(self):
        return (scan.Scan & self.item).fetch1("treadmill_times")

    @rowproperty
    def values(self):
        return (pipe_tread.Treadmill & self.item).fetch1("treadmill_vel")

    @rowproperty
    def homogeneous(self):
        return True


# -- Trace --


@keys
class Trace:
    """Recording Trace"""

    @property
    def keys(self):
        return [
            recording.Trace,
        ]

    @rowproperty
    def trials(self):
        """
        Returns
        -------
        foundation.recording.Trial (rows)
            all trials associated with the trace
        """
        from foundation.recording.trial import Trial, TrialSet

        # trace trials
        key = merge(self.key, recording.TraceTrials)
        return Trial & (TrialSet & key).members

    @rowproperty
    def trial_ids(self):
        """
        Returns
        -------
        Set[str]
            keys (foundation.recording.Trials)
        """
        return set(self.trials.fetch("trial_id"))


@keys
class Traces:
    """Recording Trace Set"""

    @property
    def keys(self):
        return [
            recording.TraceSet & "members > 0",
        ]

    @rowproperty
    def trials(self):
        """
        Returns
        -------
        foundation.recording.Trial (rows)
            all trials associated with the trace set
        """
        from foundation.recording.trace import TraceSet
        from foundation.recording.trial import Trial, TrialSet

        # trace set trials
        key = (TraceSet & self.item).members
        key = merge(key, recording.TraceTrials)
        return Trial & (TrialSet & key).members

    @rowproperty
    def trial_ids(self):
        """
        Returns
        -------
        Set[str]
            keys (foundation.recording.Trials)
        """
        return set(self.trials.fetch("trial_id"))
