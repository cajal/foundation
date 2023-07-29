import numpy as np
from foundation.virtual import utility
from foundation.recording.trial import Trial, TrialSet
from foundation.recording.trace import Trace, TraceSet
from foundation.schemas import recording as schema


@schema.computed
class ResampledTrialTemp:
    definition = """
    -> Trial
    -> utility.Rate
    ---
    index       : blob@external    # [samples]
    """

    @property
    def key_source(self):
        return ResampledTrial.proj()

    def make(self, key):
        i = (ResampledTrial & key).fetch1("index")

        self.insert1(dict(key, index=np.load(i)))


@schema.computed
class ResampledTracesTemp:
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

    @property
    def key_source(self):
        return ResampledTraces.proj()

    def make(self, key):
        t, f = (ResampledTraces & key).fetch1("traces", "finite")

        self.insert1(dict(key, traces=np.load(t), finite=f))


@schema.computed
class ResampledTrial:
    definition = """
    -> Trial
    -> utility.Rate
    ---
    index       : blob@external     # [samples]
    """

    @property
    def key_source(self):
        return ResampledTrialTemp.proj()

    def make(self, key):
        i = (ResampledTrialTemp & key).fetch1("index")

        self.insert1(dict(key, index=i))

    # def make(self, key):
    #     from foundation.recording.compute_trial import ResampledTrial

    #     # resampled video frame indices
    #     index = (ResampledTrial & key).flip_index

    #     # save file
    #     filepath = self.createpath(key, "index", "npy")
    #     np.save(filepath, index)

    #     # insert key
    #     self.insert1(dict(key, index=filepath))


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

    @property
    def key_source(self):
        return ResampledTracesTemp.proj()

    def make(self, key):
        t, f = (ResampledTracesTemp & key).fetch1("traces", "finite")

        self.insert1(dict(key, traces=t, finite=f))

    # def make(self, key):
    #     from foundation.recording.compute_trace import ResampledTraces

    #     # resampled traces
    #     traces = (ResampledTraces & key).trial(trial_id=key["trial_id"])

    #     # trace values finite
    #     finite = np.isfinite(traces).all()

    #     # save file
    #     filepath = self.createpath(key, "traces", "npy")
    #     np.save(filepath, traces)

    #     # insert key
    #     self.insert1(dict(key, traces=filepath, finite=bool(finite)))
