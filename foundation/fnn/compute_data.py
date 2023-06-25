import numpy as np
import pandas as pd
from djutils import keys, merge, rowproperty, rowmethod, cache_rowproperty, U
from foundation.utils import tqdm, disable_tqdm
from foundation.virtual import stimulus, recording, fnn


# ----------------------------- Network Data -----------------------------

# -- Network Data Base --


class NetworkData:
    """Network Data"""

    @rowproperty
    def sizes(self):
        """
        Returns
        -------
        int
            stimulus channels
        int
            perspective features
        int
            modulation features
        int
            number of units
        """
        raise NotImplementedError()

    @rowproperty
    def timing(self):
        """
        Returns
        -------
        float
            sampling period (seconds)
        float
            response offset (seconds)
        """
        raise NotImplementedError()

    @rowproperty
    def dataset(self):
        """
        Returns
        -------
        fnn.data.Dataset
            network dataset
        """
        raise NotImplementedError()

    @rowmethod
    def visual_inputs(self, video_id, trial_perspectives=True, trial_modulations=True, trial_filterset_id=None):
        """
        Parameters
        ----------
        video_id : str
            key (foundation.stimulus.video.Video)
        trial_perspectives : bool
            True (return trial perspectives) | False (return None)
        trial_modulations : bool
            True (return trial modulations) | False (return None)
        trial_filterset_id : str | None
            key (foundation.recording.TrialFilterSet) | None (no trial filtering beyond video selection)

        Returns
        -------
        Iterable[3D array]
            video frames -- [height, width, channels] x samples -- dtype=uint8
        Iterable[2D array] | None
            trial perspectives -- [trials, perspectives] x samples -- dtype=float-like ---OR--- None
        Iterable[2D array] | None
            trial modulations -- [trials, modulations] x samples -- dtype=float-like ---OR--- None
        List[str] | None
            list of trial_ids -- key (foundation.recording.trial.Trial) -- ordered by trial start ---OR--- None
        """
        raise NotImplementedError()


# -- Network Data Types --


@keys
class VisualScan(NetworkData):
    """Visual Scan Data"""

    @property
    def key_list(self):
        return [
            fnn.VisualScan,
        ]

    @rowproperty
    def key_video(self):
        return (fnn.Spec.VisualSpec & self.key).proj("height", "width", "resize_id", "rate_id").fetch1()

    def _key_traces(self, datatype):
        keymap = {f"{_}_id": f"{datatype}_{_}_id" for _ in ["resample", "offset", "standardize"]}
        key = (fnn.Spec.VisualSpec & self.key).proj("rate_id", **keymap)
        return key * recording.ScanTrials * self.key

    @rowproperty
    def key_perspective(self):
        return (self._key_traces("perspective") * recording.ScanVisualPerspectives).fetch1()

    @rowproperty
    def key_modulation(self):
        return (self._key_traces("modulation") * recording.ScanVisualModulations).fetch1()

    @rowproperty
    def key_unit(self):
        return (self._key_traces("unit") * recording.ScanUnits).fetch1()

    @rowproperty
    def trials(self):
        from foundation.recording.trial import Trial, TrialSet

        key = merge(recording.ScanRecording & self.key)
        return Trial & (TrialSet & key).members

    @rowproperty
    def sizes(self):
        # stimuli
        videos = merge(self.trials, recording.TrialVideo, stimulus.VideoInfo)
        sizes = [(U("channels") & videos).fetch1("channels")]

        # perspectives
        sizes += [(recording.TraceSet & self.key_perspective).fetch1("members")]

        # modulations
        sizes += [(recording.TraceSet & self.key_modulation).fetch1("members")]

        # units
        sizes += [(recording.TraceSet & self.key_unit).fetch1("members")]

        return tuple(sizes)

    @rowproperty
    def timing(self):
        from foundation.utility.resample import Rate, Offset

        # repsonse key
        key = self.key_unit

        # sampling period
        period = (Rate & key).link.period

        # response offset
        offset = (Offset & key).link.offset

        return period, offset

    @rowproperty
    def dataset(self):
        from foundation.recording.compute_trace import StandardizedTraces
        from fnn.data import NpyFile, Dataset

        # data keys
        key_v = self.key_video
        key_p = self.key_perspective
        key_m = self.key_modulation
        key_u = self.key_unit

        # data transforms
        transform_p = (StandardizedTraces & key_p).transform
        transform_m = (StandardizedTraces & key_m).transform
        transform_u = (StandardizedTraces & key_u).transform

        # data lists
        stimuli, perspectives, modulations, units = [], [], [], []

        # data tiers
        training_tier, validation_tier = self.key.fetch1("training_tier", "validation_tier")
        tier_keys = [{"tier_index": index} for index in [training_tier, validation_tier]]

        # trials
        trials = merge(
            recording.ScanTrials & self.key,
            recording.TrialTier & self.key & tier_keys,
            recording.TrialBounds,
            recording.TrialSamples,
            recording.TrialVideo,
            stimulus.Video,
        )
        trials, ids, tiers, samples = trials.fetch("KEY", "trial_id", "tier_index", "samples", order_by="start")

        # load trials
        for trial in tqdm(trials, desc="Trials"):

            # stimuli
            video, index = (stimulus.ResizedVideo * recording.ResampledTrial & trial & key_v).fetch1("video", "index")
            trial_stimuli = np.load(video)[np.load(index)].astype(np.uint8)

            # perspectives
            traces = (recording.ResampledTraces & trial & key_p).fetch1("traces")
            trial_perspectives = transform_p(np.load(traces)).astype(np.float32)

            # modulations
            traces = (recording.ResampledTraces & trial & key_m).fetch1("traces")
            trial_modulations = transform_m(np.load(traces)).astype(np.float32)

            # units
            traces = (recording.ResampledTraces & trial & key_u).fetch1("traces")
            trial_units = transform_u(np.load(traces)).astype(np.float32)

            # append
            stimuli.append(NpyFile(trial_stimuli))
            perspectives.append(NpyFile(trial_perspectives))
            modulations.append(NpyFile(trial_modulations))
            units.append(NpyFile(trial_units))

        # dataset
        data = {
            "training": tiers == training_tier,
            "samples": samples,
            "stimuli": stimuli,
            "perspectives": perspectives,
            "modulations": modulations,
            "units": units,
        }
        data = pd.DataFrame(data, index=pd.Index(ids, name="trial_id"))
        return Dataset(data)

    @rowmethod
    def visual_inputs(self, video_id, trial_perspectives=True, trial_modulations=True, trial_filterset_id=None):
        from foundation.utils.resample import flip_index, truncate
        from foundation.utility.resample import Rate
        from foundation.stimulus.compute_video import ResizedVideo
        from foundation.recording.compute_trace import StandardizedTraces, ResampledTraces
        from foundation.recording.trial import Trial, TrialVideo, TrialBounds, TrialFilterSet

        # data keys
        key_v = self.key_video
        key_p = self.key_perspective
        key_m = self.key_modulation

        # resized video
        key = self.key_video
        video = (ResizedVideo & key_v & {"video_id": video_id}).video

        # resampled video
        time_scale = (recording.ScanVideoTimeScale & self.key).fetch1("time_scale")
        period = (Rate & key_v).link.period
        index = flip_index(video.times * time_scale, period)
        video = video.array[index]

        # neither perspectives nor modulations requested
        if not trial_perspectives and not trial_modulations:
            return video, None, None, None

        # all trials
        trials = self.trials

        # filtered trials
        if trial_filterset_id is not None:
            trials = (TrialFilterSet & {"trial_filterset_id": trial_filterset_id}).filter(trials)

        # video trials
        trials = merge(trials, TrialVideo, TrialBounds) & {"video_id": video_id}
        trials = trials.fetch("trial_id", order_by="start").tolist()

        # no trials
        if not trials:
            return video, None, None, None

        # perspectives and modulations
        perspectives_modulations = []

        for key, requested, desc in [
            [key_p, trial_perspectives, "Perspectives"],
            [key_m, trial_modulations, "Modulations"],
        ]:
            if requested:
                # transform and resampler
                transform = (StandardizedTraces & key).transform
                resampler = ResampledTraces & key

                # resample and transform traces
                traces = (resampler.trial(trial_id=trial) for trial in tqdm(trials, desc=desc))
                with cache_rowproperty(), disable_tqdm():
                    traces = np.stack(truncate(*map(transform, traces), tolerance=1), axis=1)

                # verify length
                assert len(traces) == len(video)

                # append traces
                perspectives_modulations.append(traces)

            else:
                # append none
                perspectives_modulations.append(None)

        return video, *perspectives_modulations, trials
