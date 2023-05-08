import os
import numpy as np
from datajoint import U
from djutils import Filepath
from operator import add
from functools import reduce
from foundation.virtual import utility
from foundation.recording.trial import Trial, TrialSet
from foundation.recording.trace import Trace, TraceSet
from foundation.recording.scan import ScanTrials, ScanUnits, ScanModulations, ScanPerspectives
from foundation.schemas import recording as schema


@schema.computed
class ResampledVideo(Filepath):
    definition = """
    -> Trial
    -> utility.Rate
    ---
    index       : filepath@scratch09    # npy file, [samples]
    samples     : int unsigned          # number of samples
    """

    def make(self, key):
        from foundation.recording.compute import ResampleVideo

        # resampled video frame indices
        index = (ResampleVideo & key).index

        # save file
        filepath = self.createpath(key, "index", "npy")
        np.save(filepath, index)

        # insert key
        self.insert1(dict(key, index=filepath, samples=len(index)))


@schema.computed
class ResampledTraces(Filepath):
    definition = """
    -> TraceSet
    -> Trial
    -> utility.Rate
    -> utility.Offset
    -> utility.Resample
    ---
    traces      : filepath@scratch09    # npy file, [samples, traces]
    finite      : bool                  # all values finite
    """

    def make(self, key):
        from foundation.recording.compute import ResampleTraces

        # resamples traces
        traces = (ResampleTraces & key).trial

        # trace values finite
        finite = np.isfinite(traces).all()

        # save file
        filepath = self.createpath(key, "traces", "npy")
        np.save(filepath, traces)

        # insert key
        self.insert1(dict(key, traces=filepath, finite=bool(finite)))
