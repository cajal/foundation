import numpy as np
from tqdm import tqdm
from djutils import keys, rowproperty, rowmethod
from foundation.virtual import utility, recording


# ----------------------------- Resample -----------------------------


@keys
class ResampledTrial:
    """Resample Trial"""

    @property
    def keys(self):
        return [
            recording.Trial,
            utility.Rate,
        ]

    @rowproperty
    def samples(self):
        """
        Returns
        -------
        int
            number of resampled time points
        """
        from foundation.utils.resample import samples
        from foundation.utility.resample import Rate

        # trial timing
        start, end = (recording.TrialBounds & self.item).fetch1("start", "end")

        # resampling period
        period = (Rate & self.item).link.period

        # trial samples
        return samples(start, end, period)

    @rowproperty
    def flip_index(self):
        """
        Returns
        -------
        1D array
            stimulus flip index for each of the resampled time points
        """
        from foundation.utils.resample import flip_index
        from foundation.utility.resample import Rate
        from foundation.recording.trial import Trial

        # trial flip times
        flips = (Trial & self.item).link.compute.flip_times

        # resampling period
        period = (Rate & self.item).link.period

        # start time
        start = (recording.TrialBounds & self.item).fetch1("start")

        # interpolated flip index
        return flip_index(flips - start, period)


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
        1D array
            [samples] -- resampled trace
        """
        from foundation.recording.compute.trace import Trace

        # verify trial_id
        assert trial_id in (Trace & self.item).trial_ids, "Invalid trial_id"

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
        from foundation.recording.compute.trace import Trace

        # verify trial_ids
        assert not set(trial_ids) - (Trace & self.item).trial_ids, "Invalid trial_ids"

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
        from foundation.recording.compute.trace import Traces

        # verify trial_id
        assert trial_id in (Traces & self.item).trial_ids, "Invalid trial_id"

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
        from foundation.recording.compute.trace import Traces

        # verify trial_ids
        assert not set(trial_ids) - (Traces & self.item).trial_ids, "Invalid trial_ids"

        # trace resamplers
        resamplers = self.resamplers

        for trial_id in trial_ids:
            # trial start and end times
            start, end = (recording.TrialBounds & {"trial_id": trial_id}).fetch1("start", "end")

            # resampled traces
            yield np.stack([r(start, end) for r in resamplers], axis=1)
