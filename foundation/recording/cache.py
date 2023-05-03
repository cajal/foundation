import os
import numpy as np
import datajoint as dj
from djutils import merge, Files
from tempfile import TemporaryDirectory
from pandas.testing import assert_series_equal
from operator import add
from functools import reduce
from tqdm import tqdm
from foundation.recording import trial, trace
from foundation.utility import resample
from foundation.schemas import recording as schema


@schema.computed
class ResampledVideo(Files):
    store = "scratch09"
    definition = """
    -> trial.TrialLink
    -> resample.RateLink
    ---
    index       : filepath@scratch09    # npy file, [samples]
    samples     : int unsigned          # number of samples
    """

    def make(self, key):
        # resampling rate
        rate_key = resample.RateLink & key

        # resampled video frame indices
        index = (trial.TrialLink & key).resampled_video(rate_key)

        # save file
        file = os.path.join(self.tuple_dir(key, create=True), "index.npy")
        np.save(file, index)

        # insert key
        self.insert1(dict(key, index=file, samples=len(index)))


@schema.computed
class ResampledTraces(Files):
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
    def scan_keys(self):
        from foundation.recording.scan import (
            ScanTrialSet,
            ScanUnitSet,
            ScanModulationSet,
            ScanPerspectiveSet,
        )

        return [
            trial.TrialSet.Member * ScanTrialSet * ScanUnitSet,
            trial.TrialSet.Member * ScanTrialSet * ScanPerspectiveSet,
            trial.TrialSet.Member * ScanTrialSet * ScanModulationSet,
        ]

    @property
    def keys(self):
        keys = self.scan_keys
        keys = reduce(add, [dj.U("traces_id", "trial_id") & key for key in keys])
        keys = keys & (trace.TraceSet & "members > 0")
        keys = keys * (resample.RateLink * resample.OffsetLink * resample.ResampleLink).proj()
        return keys - self

    @property
    def key_source(self):
        key = dj.U("traces_id", "rate_id", "offset_id", "resample_id")
        key = key.aggr(self.keys, trial_id="min(trial_id)")
        return key * trial.TrialLink.proj()

    def make(self, key):
        # trial set
        key.pop("trial_id")
        trial_keys = self.keys & key
        trial_keys = trial.TrialLink & trial_keys

        # resampling method
        rate_key = resample.RateLink & key
        offset_key = resample.OffsetLink & key
        resample_key = resample.ResampleLink & key

        def sample(trace_key):
            link = trace.TraceLink & trace_key
            return link.resampled_trials(trial_keys, rate_key, offset_key, resample_key)

        # trace set, ordered by member_id
        trace_keys = (trace.TraceSet & key).members
        trace_keys = trace_keys.fetch("KEY", order_by="member_id")

        # sample first trace
        s = sample(trace_keys[0])
        n = s.apply(lambda x: x.size)

        with TemporaryDirectory() as tmpdir:

            # temporary memmap
            memmap = np.memmap(
                filename=os.path.join(tmpdir, "traces.dat"),
                shape=(len(trace_keys), np.concatenate(s).size),
                dtype=np.float32,
                mode="w+",
            )

            # write first trace to memmap
            memmap[0] = np.concatenate(s).astype(np.float32)
            memmap.flush()

            # write other traces to memmap
            for i, trace_key in enumerate(tqdm(trace_keys[1:], desc="Traces")):

                # sample trace
                _s = sample(trace_key)
                _n = _s.apply(lambda x: x.size)

                # ensure trial ids and sample sizes match
                assert_series_equal(n, _n)

                # write to memmap
                memmap[i + 1] = np.concatenate(_s).astype(np.float32)
                memmap.flush()

            # read traces from memmap and save to file
            j = 0
            for trial_id, trial_n in tqdm(n.items(), desc="Trials", total=len(n)):

                # trace values
                _traces = memmap[:, j : j + trial_n].T
                _finite = bool(np.isfinite(_traces).all())

                # save to file
                _key = dict(key, trial_id=trial_id)
                _dir = self.tuple_dir(_key, create=True)
                _file = os.path.join(_dir, "traces.npy")
                np.save(_file, _traces)

                # insert key
                self.insert1(dict(_key, traces=_file, finite=_finite))

                # memmap index
                j += trial_n
