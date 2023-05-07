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

    def make(self, key):
        from foundation.recording.compute import ResampleTraces

        # resamples traces
        traces = (ResampleTraces & key).traces

        # trace values finite
        finite = np.isfinite(traces).all()

        # save file
        file = os.path.join(self.tuple_dir(key, create=True), "traces.npy")
        np.save(file, traces)

        # insert key
        self.insert1(dict(key, traces=file, finite=bool(finite)))
