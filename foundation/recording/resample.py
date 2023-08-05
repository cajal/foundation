import numpy as np
from foundation.virtual import utility
from foundation.recording.trial import Trial
from foundation.recording.trace import TraceSet
from foundation.schemas import recording as schema


# ----------------------------- Resample -----------------------------


@schema.computed
class TrialSamples:
    definition = """
    -> Trial
    -> utility.Rate
    ---
    samples     : int unsigned      # number of samples
    """

    def make(self, key):
        from foundation.recording.compute.resample import ResampledTrial

        # number of samples
        key["samples"] = (ResampledTrial & key).samples

        # insert
        self.insert1(key)


@schema.computed
class ResampledTrial:
    definition = """
    -> Trial
    -> utility.Rate
    ---
    index       : blob@external     # [samples]
    """

    def make(self, key):
        from foundation.recording.compute.resample import ResampledTrial

        # resampled flip indices
        index = (ResampledTrial & key).flip_index

        # insert
        self.insert1(dict(key, index=index))


@schema.computed
class ResampledTraces:
    definition = """
    -> TraceSet
    -> Trial
    -> utility.Resample
    -> utility.Offset
    -> utility.Rate
    ---
    traces      : blob@external     # [samples, traces]
    finite      : bool              # all values finite
    """

    def make(self, key):
        from foundation.recording.compute.resample import ResampledTraces

        # resampled traces
        traces = (ResampledTraces & key).trial(trial_id=key["trial_id"])

        # trace values finite
        finite = np.isfinite(traces).all()

        # insert
        self.insert1(dict(key, traces=traces, finite=bool(finite)))
