import numpy as np
import pandas as pd
from datajoint import U
from djutils import keys, merge, rowproperty, rowmethod
from foundation.fnn.dataset import VisualRecording
from foundation.fnn.dataspec import VisualSpec, ResampleVisual


@keys
class ResampledVisualRecording:
    """Load Preprocessed Data"""

    @property
    def key_list(self):
        return [
            VisualRecording,
            VisualSpec.ResampleVisual,
        ]

    @rowproperty
    def trials(self):
        from foundation.recording.trial import Trial, TrialSet

        key = merge(self.key, VisualRecording)
        return Trial & (TrialSet & key).members

    @rowproperty
    def trial_samples(self):
        from foundation.recording.trial import TrialSamples

        key = merge(self.key, VisualSpec.ResampleVisual)

        trials = merge(self.trials, TrialSamples & key)
        trial_id, samples = trials.fetch("trial_id", "samples", order_by="trial_id", limit=5)  # TODO

        return pd.Series(data=samples, index=pd.Index(trial_id, name="trial_id"))

    @rowproperty
    def trial_video(self):
        from foundation.recording.trial import TrialVideo
        from foundation.recording.cache import ResampledVideo
        from foundation.stimulus.cache import ResizedVideo
        from fnn.data import NpyFile

        key = merge(self.key, VisualSpec.ResampleVisual)

        trials = merge(self.trials, TrialVideo, ResizedVideo & key, ResampledVideo & key)
        trial_id, video, index = trials.fetch("trial_id", "video", "index", order_by="trial_id", limit=5)  # TODO

        data = [NpyFile(v, indexmap=np.load(i)) for v, i in zip(video, index)]
        return pd.Series(data=data, index=pd.Index(trial_id, name="trial_id"))

    @rowmethod
    def traceset_key(self, suffix="p"):
        if suffix not in ["p", "m", "u"]:
            raise ValueError("Suffix must be one of {'p', 'm', 'u'}")

        proj = {f"{k}_id": f"{k}_id_{suffix}" for k in ["traceset", "offset", "resample", "standardize"]}
        key = merge(self.key, VisualRecording, VisualSpec.ResampleVisual).proj(..., **proj)

        attrs = ["traceset_id", "trialset_id", "standardize_id", "rate_id", "offset_id", "resample_id"]
        key = U(*attrs) & key

        return key.fetch1("KEY")

    @rowmethod
    def trial_traces(self, suffix="p"):
        from foundation.recording.compute import StandardizeTraces
        from foundation.recording.cache import ResampledTraces
        from fnn.data import NpyFile

        key = self.traceset_key(suffix)

        transform = (StandardizeTraces & key).transform

        trials = merge(self.trials, ResampledTraces & key & "finite")
        trial_id, traces = trials.fetch("trial_id", "traces", order_by="trial_id", limit=5)  # TODO

        data = [NpyFile(t, transform=transform) for t in traces]
        return pd.Series(data=data, index=pd.Index(trial_id, name="trial_id"))

    @rowproperty
    def visual_dataset(self):
        from fnn.data import Dataset

        data = [
            self.trial_samples.rename("samples"),
            self.trial_video.rename("stimuli"),
            self.trial_traces("p").rename("perspectives"),
            self.trial_traces("m").rename("modulations"),
            self.trial_traces("u").rename("units"),
        ]
        df = pd.concat(data, axis=1, join="outer")
        assert not df.isnull().values.any()

        return Dataset(df)
