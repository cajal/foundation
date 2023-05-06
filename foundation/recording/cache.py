import os
import numpy as np
from datajoint import U
from djutils import Filepath
from operator import add
from functools import reduce
from foundation.virtual import utility
from foundation.recording.trial import TrialLink, TrialSet
from foundation.recording.trace import TraceLink, TraceSet
from foundation.recording.scan import ScanTrials, ScanUnits, ScanModulations, ScanPerspectives
from foundation.schemas import recording as schema


@schema.computed
class ResampledVideo(Filepath):
    store = "scratch09"
    definition = """
    -> TrialLink
    -> utility.RateLink
    ---
    index       : filepath@scratch09    # npy file, [samples]
    samples     : int unsigned          # number of samples
    """

    def make(self, key):
        from foundation.recording.compute import ResampleVideo

        # resampled video frame indices
        index = (ResampleVideo & key).index

        # save file
        file = os.path.join(self.tuple_dir(key, create=True), "index.npy")
        np.save(file, index)

        # insert key
        self.insert1(dict(key, index=file, samples=len(index)))


@schema.computed
class ResampledTraces(Filepath):
    store = "scratch09"
    definition = """
    -> TraceSet
    -> TrialLink
    -> utility.RateLink
    -> utility.OffsetLink
    -> utility.ResampleLink
    ---
    traces      : filepath@scratch09    # npy file, [samples, traces]
    finite      : bool                  # all values finite
    """

    @property
    def keys(self):
        keys = [
            ScanUnits * ScanTrials * TrialSet.Link,
            ScanPerspectives * ScanTrials * TrialSet.Link,
            ScanModulations * ScanTrials * TrialSet.Link,
        ]
        keys = reduce(add, [U("traces_id", "trial_id") & key for key in keys])
        keys = keys * utility.RateLink.proj() * utility.OffsetLink.proj() * utility.ResampleLink.proj()
        return keys - self

    @property
    def key_source(self):
        key = U("traces_id", "rate_id", "offset_id", "resample_id")
        key = key.aggr(self.keys, trial_id="min(trial_id)")

        return key * TrialLink.proj()

    def make(self, key):
        from foundation.recording.compute import ResampleTraces

        # trials
        key.pop("trial_id")
        trials = TrialLink & (self.keys & key)

        # reampled traces for each trial
        for trial_id, traces in (ResampleTraces & key & trials).trials:

            # trial key
            _key = dict(key, trial_id=trial_id)

            # trace values finite
            finite = np.isfinite(traces).all()

            # save file
            file = os.path.join(self.tuple_dir(_key, create=True), "traces.npy")
            np.save(file, traces)

            # insert key
            self.insert1(dict(_key, traces=file, finite=bool(finite)))
