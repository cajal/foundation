import os
import numpy as np
import datajoint as dj
from djutils import merge, Files
from tempfile import TemporaryDirectory
from operator import add
from functools import reduce
from tqdm import tqdm
from foundation.recording import trial, trace
from foundation.utility import resample
from foundation.schemas import recording as schema


@schema.computed
class TrialTraces(Files):
    store = "scratch09"
    definition = """
    -> trace.TraceSet
    -> resample.RateLink
    -> resample.OffsetLink
    -> resample.ResampleLink
    -> trial.TrialLink
    ---
    traces      : filepath@scratch09    # npy file, [samples, traces]
    finite      : bool                  # all values finite
    """

    @property
    def keys(self):
        from foundation.recording.scan import (
            ScanTrials,
            ScanResponses,
            ScanModulation,
            ScanPerspective,
        )

        keys = [
            trial.TrialSet.Member * ScanTrials * ScanResponses,
            trial.TrialSet.Member * ScanTrials * ScanModulation,
            trial.TrialSet.Member * ScanTrials * ScanPerspective,
        ]
        keys = reduce(add, [dj.U("traces_id", "trial_id") & key for key in keys])

        return keys * (resample.RateLink * resample.OffsetLink * resample.ResampleLink).proj()

    @property
    def key_source(self):
        return dj.U("traces_id", "rate_id", "offset_id", "resample_id") & self.keys

    def make(self, key):
        # fetch all trials for trace set, ordered by trial_id
        trials = self.keys & key
        trials = merge(trials, trial.TrialSamples & key)
        trial_ids, samples = trials.fetch("trial_id", "samples", order_by="trial_id")

        # fetch all traces for trace set, orderd by member_id of trace set
        traces = (trace.TraceSet & key).members
        traces = merge(traces, trace.TraceSamples & key)
        trace_keys = traces.fetch("KEY", order_by="member_id")

        with TemporaryDirectory() as tmpdir:

            # temporary memmap
            memmap = np.memmap(
                filename=os.path.join(tmpdir, "traces.dat"),
                shape=(len(trace_keys), sum(samples)),
                dtype=np.float32,
                mode="w+",
            )

            # write traces to memmap
            for i, trace_key in enumerate(tqdm(trace_keys, desc="Traces")):

                j = 0
                df = (trace.TraceSamples & trace_key).trials.loc[trial_ids]

                for tup, trial_id, trial_n in zip(df.itertuples(), trial_ids, samples):

                    assert trial_id == tup.Index
                    assert trial_n == tup.trace.size

                    memmap[i, j : j + trial_n] = tup.trace.astype(np.float32)
                    j += trial_n

                memmap.flush()

            # read traces from memmap and save to file
            j = 0
            for trial_id, trial_n in zip(trial_ids, tqdm(samples, desc="Trials")):

                _traces = memmap[j : j + trial_n]
                _finite = bool(np.isfinite(_traces).all())

                _key = dict(key, trial_id=trial_id)
                _dir = self.tuple_dir(_key, create=True)
                _file = os.path.join(_dir, "traces.npy")

                np.save(_file, _traces)
                self.insert1(dict(_key, traces=_file, finite=_finite))

                j += trial_n
