import os
import numpy as np
from datajoint import U
from djutils import Filepath
from operator import add
from functools import reduce
from foundation.virtual import utility
from foundation.recording.trial import Trial, TrialSet
from foundation.recording.trace import Trace, TraceSet
from foundation.recording.scan import ScanTrials, ScanUnits, ScanVisualModulations, ScanVisualPerspectives
from foundation.schemas import recording as schema


@schema.computed
class ResampledVideo(Filepath):
    definition = """
    -> Trial
    -> utility.Rate
    ---
    index       : filepath@scratch09    # npy file, [samples]
    """

    def make(self, key):
        from foundation.recording.compute import ResampleTrial

        # resampled video frame indices
        index = (ResampleTrial & key).video_index

        # save file
        filepath = self.createpath(key, "index", "npy")
        np.save(filepath, index)

        # insert key
        self.insert1(dict(key, index=filepath))


@schema.computed
class ResampledTraces(Filepath):
    definition = """
    -> TraceSet
    -> Trial
    -> utility.Resample
    -> utility.Offset
    -> utility.Rate
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
