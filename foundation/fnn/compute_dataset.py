import numpy as np
import pandas as pd
from datajoint import U
from djutils import keys, merge, rowproperty, rowmethod, cache_rowproperty
from foundation.utils import logger, tqdm, disable_tqdm
from foundation.virtual import utility, stimulus, recording, fnn


# ----------------------------- Dataset -----------------------------


@keys
class VisualScan:
    """Visual Scan"""

    @property
    def key_list(self):
        return [fnn.VisualScan]

    @rowproperty
    def stimuli_key(self):
        return merge(
            self.key.proj(spec_id="stimuli_id"),
            fnn.Spec.VideoSpec,
        ).fetch1()

    @rowproperty
    def perspectives_key(self):
        return merge(
            self.key.proj(spec_id="perspectives_id"),
            fnn.Spec.TraceSpec,
            recording.ScanVisualPerspectives,
            recording.ScanTrials,
        ).fetch1()

    @rowproperty
    def modulations_key(self):
        return merge(
            self.key.proj(spec_id="modulations_id"),
            fnn.Spec.TraceSpec,
            recording.ScanVisualModulations,
            recording.ScanTrials,
        ).fetch1()

    @rowproperty
    def units_key(self):
        return merge(
            self.key.proj(spec_id="units_id"),
            fnn.Spec.TraceSpec,
            recording.ScanUnits,
            recording.ScanTrials,
        ).fetch1()

    @rowproperty
    def trials(self):
        from foundation.recording.trial import Trial, TrialSet

        key = merge(self.key, recording.ScanTrials).fetch1()

        return Trial & (TrialSet & key).members

    @rowproperty
    def samples(self):
        from foundation.recording.trial import TrialSamples

        trials = merge(
            self.trials,
            TrialSamples & self.key,
        )
        trial_id, samples = trials.fetch("trial_id", "samples", order_by="trial_id")
        index = pd.Index(trial_id, name="trial_id")

        return pd.Series(data=samples, index=index)

    @rowproperty
    def stimuli(self):
        from fnn.data import NpyFile

        key = self.stimuli_key
        trials = merge(
            self.trials,
            recording.TrialVideo,
            stimulus.ResizedVideo & key,
            recording.ResampledTrial & key,
        )
        trial_id, video, imap = trials.fetch("trial_id", "video", "index", order_by="trial_id")
        index = pd.Index(trial_id, name="trial_id")
        data = [NpyFile(v, indexmap=np.load(i), dtype=np.uint8) for v, i in zip(video, tqdm(imap, desc="Video"))]

        return pd.Series(data=data, index=index)

    @rowmethod
    def _traces(self, key):
        from foundation.recording.compute_trace import StandardTraces
        from fnn.data import NpyFile

        transform = (StandardTraces & key).transform
        trials = merge(
            self.trials,
            recording.ResampledTraces & key,
        )
        trial_id, traces = trials.fetch("trial_id", "traces", order_by="trial_id")
        index = pd.Index(trial_id, name="trial_id")
        data = [NpyFile(t, transform=transform, dtype=np.float32) for t in tqdm(traces, desc="Traces")]

        return pd.Series(data=data, index=index)

    @rowproperty
    def dataset(self):
        from fnn.data import Dataset

        data = [
            self.samples.rename("samples"),
            self.stimuli.rename("stimuli"),
            self._traces(self.perspectives_key).rename("perspectives"),
            self._traces(self.modulations_key).rename("modulations"),
            self._traces(self.units_key).rename("units"),
        ]
        df = pd.concat(data, axis=1, join="outer")
        assert not df.isnull().values.any()

        return Dataset(df)
