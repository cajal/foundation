import os
import numpy as np
import datajoint as dj
from djutils import Files
from tempfile import TemporaryDirectory
from tqdm import tqdm
from operator import add
from functools import reduce
from pandas.testing import assert_series_equal
from foundation.utility.resample import RateLink, OffsetLink, ResampleLink
from foundation.recording.trial import TrialLink, TrialSet
from foundation.recording.trace import TraceLink, TraceSet
from foundation.recording.resample import ResampledTrialVideo, ResampledTraceTrials
from foundation.schemas import recording as schema


@schema.computed
class ResampledVideoIndex(Files):
    store = "scratch09"
    definition = """
    -> RateLink
    -> TrialLink
    ---
    index       : filepath@scratch09    # npy file, [samples]
    samples     : int unsigned          # number of samples
    """

    def make(self, key):
        # resampled video frame indices
        index = (ResampledTrialVideo & key).index

        # save file
        file = os.path.join(self.tuple_dir(key, create=True), "index.npy")
        np.save(file, index)

        # insert key
        self.insert1(dict(key, index=file, samples=len(index)))


@schema.computed
class ResampledTraces(Files):
    store = "scratch09"
    definition = """
    -> RateLink
    -> OffsetLink
    -> ResampleLink
    -> TraceSet
    -> TrialLink
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
            TrialSet.Member * ScanTrialSet * ScanUnitSet,
            TrialSet.Member * ScanTrialSet * ScanPerspectiveSet,
            TrialSet.Member * ScanTrialSet * ScanModulationSet,
        ]

    @property
    def keys(self):
        keys = self.scan_keys
        keys = reduce(add, [dj.U("traces_id", "trial_id") & key for key in keys])
        keys = keys & (TraceSet & "members > 0")
        keys = keys * (RateLink * OffsetLink * ResampleLink).proj()
        return keys - self

    @property
    def key_source(self):
        key = dj.U("traces_id", "rate_id", "offset_id", "resample_id")
        key = key.aggr(self.keys, trial_id="min(trial_id)")
        return key * TrialLink.proj()

    def make(self, key):
        # trace keys, ordered by member_id
        trace_keys = (TraceSet & key).members.fetch("KEY", order_by="member_id")

        # trial keys
        key.pop("trial_id")
        trial_keys = TrialLink & (self.keys & key)

        # resample trace
        def samples(trace_key):
            return (ResampledTraceTrials & key & trial_keys & trace_key).samples

        # resample first trace
        s = samples(trace_keys[0])
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

                # resample trace
                _s = samples(trace_key)
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
