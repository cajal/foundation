import numpy as np
from djutils import keys, merge, rowmethod, rowproperty
from foundation.utils import tqdm
from foundation.virtual.bridge import pipe_fuse, pipe_shared, pipe_tread, resolve_pipe
from foundation.virtual import utility, stimulus, scan, recording


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
    def valid_trials(self):
        """
        Returns
        -------
        foundation.recording.Trial (rows)
            valid trials
        """
        from foundation.recording.trial import Trial, TrialSet

        # trace trials
        key = merge(self.key, recording.TraceTrials)
        return Trial & (TrialSet & key).members


@keys
class Traces:
    """Recording Trace Set"""

    @property
    def keys(self):
        return [
            recording.TraceSet & "members > 0",
        ]

    @rowproperty
    def valid_trials(self):
        """
        Returns
        -------
        foundation.recording.Trial (rows)
            valid trials
        """
        from foundation.recording.trace import TraceSet
        from foundation.recording.trial import Trial, TrialSet

        # trace set trials
        key = (TraceSet & self.item).members
        key = merge(key, recording.TraceTrials)
        return Trial & (TrialSet & key).members


# ----------------------------- Resampling -----------------------------


@keys
class ResampledTrace:
    """Resampled Trace"""

    @property
    def keys(self):
        return [
            recording.Trace,
            utility.Resample,
            utility.Offset,
            utility.Rate,
        ]

    @rowproperty
    def resampler(self):
        """
        Returns
        -------
        foundation.utils.resample.Resample
            callable, trace resampler
        """
        from foundation.utility.resample import Rate, Offset, Resample
        from foundation.recording.trace import Trace

        # resampling period, offset, method
        period = (Rate & self.item).link.period
        offset = (Offset & self.item).link.offset
        resample = (Resample & self.item).link.resample

        # trace resampler
        trace = (Trace & self.item).link.compute
        return resample(times=trace.times, values=trace.values, target_period=period, target_offset=offset)

    @rowmethod
    def trial(self, trial_id):
        """
        Parameters
        ----------
        trial_id : str
            key (foundation.recording.trial.Trial)

        Returns
        -------
        1D array -- [samples]
            resampled trace
        """
        # verify trial_id
        valid_ids = (Trace & self.item).valid_trials.fetch("trial_id")
        assert trial_id in valid_ids, "Invalid trial_id"

        # trial start and end times
        start, end = (recording.TrialBounds & {"trial_id": trial_id}).fetch1("start", "end")

        # resampled trace
        return self.resampler(start, end)

    @rowmethod
    def trials(self, trial_ids):
        """
        Parameters
        ----------
        trial_id : Sequence[str]
            sequence of keys (foundation.recording.trial.Trial)

        Yields
        ------
        1D array
            [samples] -- resampled trace
        """
        # verify trial_ids
        valid_ids = (Trace & self.item).valid_trials.fetch("trial_id")
        assert not set(trial_ids) - set(valid_ids), "Invalid trial_ids"

        # trace resampler
        resampler = self.resampler

        for trial_id in trial_ids:
            # trial start and end times
            start, end = (recording.TrialBounds & {"trial_id": trial_id}).fetch1("start", "end")

            # resampled trace
            yield resampler(start, end)


@keys
class ResampledTraces:
    """Resampled Trace Set"""

    @property
    def keys(self):
        return [
            recording.TraceSet,
            utility.Resample,
            utility.Offset,
            utility.Rate,
        ]

    @rowproperty
    def resamplers(self):
        """
        Returns
        -------
        tuple[foundation.utils.resample.Resample]
            tuple of callables, trace resamplers, ordered by traceset_index
        """
        from foundation.recording.trace import TraceSet

        # trace set
        traces = (TraceSet & self.item).members
        traces = traces.fetch("trace_id", order_by="traceset_index", as_dict=True)
        traces = tqdm(traces, desc="Traces")

        # trace resamplers
        resamplers = []
        for trace in traces:
            resampler = (ResampledTrace & trace & self.item).resampler
            resamplers.append(resampler)

        return tuple(resamplers)

    @rowmethod
    def trial(self, trial_id):
        """
        Parameters
        ----------
        trial_id : str
            key (foundation.recording.trial.Trial)

        Returns
        -------
        2D array
            [samples, traces] -- resampled traces, ordered by traceset index
        """
        # verify trial_id
        valid_ids = (Traces & self.item).valid_trials.fetch("trial_id")
        assert trial_id in valid_ids, "Invalid trial_id"

        # trial start and end times
        start, end = (recording.TrialBounds & {"trial_id": trial_id}).fetch1("start", "end")

        # resampled traces
        return np.stack([r(start, end) for r in self.resamplers], axis=1)

    @rowmethod
    def trials(self, trial_ids):
        """
        Parameters
        ----------
        trial_id : Sequence[str]
            sequence of keys (foundation.recording.trial.Trial)

        Yields
        ------
        2D array
            [samples, traces] -- resampled traces (ordered by traceset index)
        """
        # verify trial_ids
        valid_ids = (Traces & self.item).valid_trials.fetch("trial_id")
        assert not set(trial_ids) - set(valid_ids), "Invalid trial_ids"

        # trace resamplers
        resamplers = self.resamplers

        for trial_id in trial_ids:
            # trial start and end times
            start, end = (recording.TrialBounds & {"trial_id": trial_id}).fetch1("start", "end")

            # resampled traces
            yield np.stack([r(start, end) for r in resamplers], axis=1)


# ----------------------------- Statistics -----------------------------


@keys
class TraceSummary:
    """Trace Summary"""

    @property
    def keys(self):
        return [
            recording.Trace,
            recording.TrialSet & "members > 0",
            utility.Summary,
            utility.Resample,
            utility.Offset,
            utility.Rate,
        ]

    @rowproperty
    def statistic(self):
        """
        Returns
        -------
        float
            trace summary statistic
        """
        from foundation.utility.stat import Summary
        from foundation.recording.trial import TrialSet

        # recording trials
        trial_ids = (TrialSet & self.item).members.fetch("trial_id", order_by="trialset_index")

        # resampled traces
        trials = (ResampledTrace & self.item).trials(trial_ids)
        trials = np.concatenate(list(trials))

        # summary statistic
        return (Summary & self.item).link.summary(trials)


# ----------------------------- Standardization -----------------------------


@keys
class StandardizedTrace:
    """Standardized Trace"""

    @property
    def keys(self):
        return [
            recording.Trace,
            recording.TrialSet & "members > 0",
            utility.Standardize,
            utility.Resample,
            utility.Offset,
            utility.Rate,
        ]

    @rowproperty
    def transform(self):
        """
        Returns
        -------
        foundation.utility.standardize.Standardize
            callable, standardizes trace
        """
        from foundation.utility.standardize import Standardize

        # homogeneous mask
        hom = merge(self.key, recording.TraceHomogeneous).fetch1("homogeneous")
        hom = [hom.astype(bool)]

        # standardization link
        stand = (Standardize & self.item).link

        # summary stats
        stat_keys = [{"summary_id": _} for _ in stand.summary_ids]
        stats = (utility.Summary & stat_keys).proj()
        stats = merge(self.key * stats, recording.TraceSummary)

        # stats dict
        summary_id, summary = stats.fetch("summary_id", "summary")
        kwargs = {k: [v] for k, v in zip(summary_id, summary)}

        # standarization transform
        return stand.standardize(homogeneous=hom, **kwargs)


@keys
class StandardizedTraces:
    """Standardized Trace Set"""

    @property
    def keys(self):
        return [
            recording.TraceSet & "members > 0",
            recording.TrialSet & "members > 0",
            utility.Standardize,
            utility.Resample,
            utility.Offset,
            utility.Rate,
        ]

    @rowproperty
    def transform(self):
        """
        Returns
        -------
        foundation.utility.standardize.Standardize
            callable, standardizes trace set
        """
        from foundation.utility.standardize import Standardize
        from foundation.recording.trace import TraceSet

        # traces
        traces = (TraceSet & self.item).members

        # homogeneous mask
        hom = merge(traces, recording.TraceHomogeneous)
        hom = hom.fetch("homogeneous", order_by="traceset_index")
        hom = hom.astype(bool)

        # standardization link
        stand = (Standardize & self.item).link

        # summary stats
        stat_keys = [{"summary_id": _} for _ in stand.summary_ids]
        stats = (utility.Summary & stat_keys).proj()
        stats = merge(self.key * traces * stats, recording.TraceSummary)

        # stats dict
        kwargs = dict()
        for skey in stat_keys:
            sid = skey["summary_id"]
            kwargs[sid] = (stats & skey).fetch("summary", order_by="traceset_index")

        # standarization transform
        return stand.standardize(homogeneous=hom, **kwargs)
