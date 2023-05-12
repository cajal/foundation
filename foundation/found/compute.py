import numpy as np
import pandas as pd
from djutils import keys, merge, rowproperty
from fnn.data import NpyFile, Data
from foundation.fnn.data import DataSet, DataSetComponents, DataSpec


# -------------- Load Data --------------


class LoadData:
    """Load Data"""

    @rowproperty
    def load(self):
        raise NotImplementedError()


@keys
class PreprocessedData(LoadData):
    """Load Preprocessed Data"""

    @property
    def key_list(self):
        return [
            DataSet,
            DataSpec.Preprocess,
        ]

    @rowproperty
    def load(self):
        from foundation.recording.trial import TrialSet, TrialSamples, TrialVideo
        from foundation.recording.compute import StandardizeTraces
        from foundation.recording.cache import ResampledVideo, ResampledTraces
        from foundation.stimulus.cache import ResizedVideo

        # dataset key
        key = merge(self.key, DataSetComponents, DataSpec.Preprocess)

        # dataset trials
        trials = (TrialSet & key).members

        # fetch samples and video
        data = merge(trials * key, TrialSamples, TrialVideo, ResampledVideo, ResizedVideo)
        trial_id, samples, video, index = data.fetch(
            "trial_id", "samples", "video", "index", order_by="trial_id", limit=5
        )  # TODO
        video = [NpyFile(v, indexmap=np.load(i), transform=transform) for v, i in zip(video, index)]

        tracecols = dict()
        for suffix, name in [["p", "perspective"], ["m", "modulation"], ["u", "unit"]]:

            # traceset key
            proj = {
                "traceset_id": f"traceset_id_{suffix}",
                "offset_id": f"offset_id_{suffix}",
                "resample_id": f"resample_id_{suffix}",
                "standardize_id": f"standardize_id_{suffix}",
            }
            _key = key.proj(..., **proj)

            # fetch traceset
            data = merge(trials * _key, ResampledTraces & "finite")
            _trial_id, traces = data.fetch("trial_id", "traces", order_by="trial_id", limit=5)  # TODO
            transform = (StandardizeTraces & _key).transform
            traces = [NpyFile(t, transform=transform) for t in traces]

            assert np.array_equal(trial_id, _trial_id)
            tracecols[name] = traces

        # data frame
        df = Data(
            data={"samples": samples, "video": video, **tracecols},
            index=pd.Index(trial_id, name="trial_id"),
        )
        # assert len(df) == (TrialSet & key).fetch1("members")

        return df
