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
        return [
            fnn.VisualScan,
        ]

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
        index = index = pd.Index(trial_id, name="trial_id")
        data = [NpyFile(t, transform=transform, dtype=np.float32) for t in tqdm(traces, desc="Traces")]

        return pd.Series(data=data, index=index)

    @rowproperty
    def trainset(self):
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

    @rowproperty
    def network_sizes(self):

        stimuli = merge(self.trials, recording.TrialVideo, stimulus.VideoInfo)
        perspectives = recording.TraceSet & self.perspectives_key
        modulations = recording.TraceSet & self.modulations_key
        units = recording.TraceSet & self.units_key

        return {
            "stimuli": (U("channels") & stimuli).fetch1("channels"),
            "perspectives": perspectives.fetch1("members"),
            "modulations": modulations.fetch1("members"),
            "units": units.fetch1("members"),
        }


# ----------------------------- Inputs -----------------------------


# @keys
# class VisualScanInputs:
#     """Visual Scan Inputs"""

#     @property
#     def key_list(self):
#         return [
#             fnn.VisualScan,
#             stimulus.Video,
#         ]

#     @rowproperty
#     def trials(self):
#         trials = (VisualScan & self.key).all_trials.proj()
#         trials = merge(trials, recording.TrialBounds, recording.TrialVideo) & self.key
#         return trials.fetch("KEY", order_by="start")

#     @rowmethod
#     def stimuli(self):
#         """
#         Yields
#         ------
#         4D array -- [batch_size, height, width, channels]
#             video frame
#         """
#         from foundation.utils.resample import truncate, flip_index
#         from foundation.utility.resample import Rate
#         from foundation.stimulus.compute import ResizeVideo
#         from foundation.recording.compute import ResampleTrial

#         # load video
#         key = (VisualScan & self.key).stimuli_key
#         video = (ResizeVideo & self.key & key).video

#         # load trials
#         trials = self.trials
#         if trials:

#             # video indexes based on trial timing
#             with cache_rowproperty():
#                 indexes = [(ResampleTrial & trial & self.key).video_index for trial in trials]

#             indexes = truncate(*indexes)
#             indexes = np.stack(indexes, axis=0)

#             if not np.diff(indexes, axis=0).any():
#                 indexes = indexes[:1]

#         else:
#             # video indexes based on expected timing
#             times = video.times
#             time_scale = merge(self.key, recording.ScanVideoTiming).fetch1("time_scale")
#             period = (Rate & self.key).link.period

#             indexes = flip_index(times * time_scale, period)[None]

#         # yield video frames
#         varray = video.array
#         for i in indexes.T:
#             yield varray[i]

#     @rowmethod
#     def perspectives(self):
#         """
#         Yields
#         ------
#         2D array -- [batch_size, traces]
#             perspective frame
#         """
#         from foundation.utils.resample import truncate
#         from foundation.recording.compute import ResampleTraces

#         # load trials
#         trials = self.trials
#         if trials:
#             trials = tqdm(trials, desc="Perspective Trials")
#         else:
#             return

#         with cache_rowproperty(), disable_tqdm():
#             # traceset key and transform
#             key = (VisualScan & self.key).perspectives_key
#             transform = (VisualScan & self.key).perspectives_transform

#             # load and transform traceset
#             inputs = ((ResampleTraces & trial & key).trial for trial in trials)
#             inputs = (transform(i) for i in truncate(*inputs))
#             inputs = np.stack(list(inputs), axis=1)

#         # yield traceset frames
#         def generate():
#             yield from inputs

#         return generate()

#     @rowmethod
#     def modulations(self):
#         """
#         Yields
#         ------
#         2D array -- [batch_size, traces]
#             modulation frame
#         """
#         from foundation.utils.resample import truncate
#         from foundation.recording.compute import ResampleTraces

#         # load trials
#         trials = self.trials
#         if trials:
#             trials = tqdm(trials, desc="Modulation Trials")
#         else:
#             return

#         with cache_rowproperty(), disable_tqdm():
#             # traceset key and transform
#             key = (VisualScan & self.key).modulations_key
#             transform = (VisualScan & self.key).modulations_transform

#             # load and transform traceset
#             inputs = ((ResampleTraces & trial & key).trial for trial in trials)
#             inputs = (transform(i) for i in truncate(*inputs))
#             inputs = np.stack(list(inputs), axis=1)

#         # yield traceset frames
#         def generate():
#             yield from inputs

#         return generate()
