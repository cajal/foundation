import numpy as np
import pandas as pd
from djutils import keys, merge, rowproperty, rowmethod, cache_rowproperty, unique, MissingError
from foundation.utils import tqdm
from foundation.virtual import stimulus, recording, fnn


# ----------------------------- Data -----------------------------

# -- Data Interface --


class DataType:
    """Data"""

    @rowproperty
    def stimuli(self):
        """
        Returns
        -------
        int
            stimulus channels
        """
        raise NotImplementedError()

    @rowproperty
    def perspectives(self):
        """
        Returns
        -------
        int
            perspective features
        """
        raise NotImplementedError()

    @rowproperty
    def modulations(self):
        """
        Returns
        -------
        int
            modulations features
        """
        raise NotImplementedError()

    @rowproperty
    def units(self):
        """
        Returns
        -------
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
        Parameters
        ----------
        video_id : str
            key (foundation.stimulus.video.Video)

        Returns
        -------
        fnn.data.Dataset
            network dataset
        """
        raise NotImplementedError()

    @rowmethod
    def visual_stimuli(self, video_id):
        """
        Returns
        -------
        4D array
            [samples, height, width, channels] (uint8) -- video frames
        """
        raise NotImplementedError()

    @rowmethod
    def visual_trial_ids(self, video_id, trial_filterset_id):
        """
        Parameters
        ----------
        video_id : str
            key (foundation.stimulus.video.Video)
        trial_filterset_id : str
            key (foundation.recording.TrialFilterSet)

        Returns
        -------
        Tuple[str]
            tuple of keys (foundation.recording.trial.Trial) -- ordered by trial start time
        """
        raise NotImplementedError()

    @rowmethod
    def trial_perspectives(self, trial_ids, perspective_index=None, use_cache=False):
        """
        Parameters
        ----------
        trial_id : Sequence[str]
            sequence of keys (foundation.recording.trial.Trial)
        perspective_index : int | None
            perspective index (specific perspective) | None (all perspectives)
        use_cache : bool
            use cache (True) | compute from scratch (False)

        Yields
        ------
        1D array | 2D array
            [samples] (specific perspective) | [samples, perspectives] (all perspectives)
        """
        raise NotImplementedError()

    @rowmethod
    def trial_modulations(self, trial_ids, modulation_index=None, use_cache=False):
        """
        Parameters
        ----------
        trial_id : Sequence[str]
            sequence of keys (foundation.recording.trial.Trial)
        modulation_index : int | None
            modulation index (specific modulation) | None (all modulations)
        use_cache : bool
            use cache (True) | compute from scratch (False)

        Yields
        ------
        1D array | 2D array
            [samples] (specific modulation) | [samples, modulations] (all modulations)
        """
        raise NotImplementedError()

    @rowmethod
    def trial_units(self, trial_ids, unit_index=None, use_cache=False):
        """
        Parameters
        ----------
        trial_id : Sequence[str]
            sequence of keys (foundation.recording.trial.Trial)
        unit_index : int | None
            unit index (specific unit) | None (all units)
        use_cache : bool
            use cache (True) | compute from scratch (False)

        Yields
        ------
        1D array | 2D array
            [samples] (specific unit) | [samples, units] (all units)
        """
        raise NotImplementedError()


# -- Data Types --


@keys
class VisualScan(DataType):
    """Visual Scan Data"""

    @property
    def keys(self):
        return [
            fnn.VisualScan,
        ]

    @rowproperty
    def key_video(self):
        return (fnn.Spec.VisualSpec & self.item).proj("height", "width", "resize_id", "rate_id").fetch1()

    def _key_traces(self, datatype):
        keymap = {f"{_}_id": f"{datatype}_{_}_id" for _ in ["resample", "offset", "standardize"]}
        return (recording.ScanTrials * fnn.Spec.VisualSpec & self.item).proj("rate_id", "trialset_id", **keymap)

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
    def stimuli(self):
        key = recording.ScanRecording & self.item
        videos = merge(recording.TrialSet.Member & key, recording.TrialVideo, stimulus.VideoInfo)
        return unique(videos, "channels")

    @rowproperty
    def perspectives(self):
        return (recording.TraceSet & self.key_perspective).fetch1("members")

    @rowproperty
    def modulations(self):
        return (recording.TraceSet & self.key_modulation).fetch1("members")

    @rowproperty
    def units(self):
        return (recording.TraceSet & self.key_unit).fetch1("members")

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
            recording.ScanTrials & self.item,
            recording.TrialTier & self.item & tier_keys,
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
    def visual_stimuli(self, video_id):
        from foundation.stimulus.compute_video import ResizedVideo
        from foundation.utility.resample import Rate
        from foundation.utils.resample import flip_index

        # video key
        key = {"video_id": video_id, **self.key_video}

        # resized video
        video = (ResizedVideo & key).video

        # time scale of recording
        time_scale = (recording.ScanVideoTimeScale & self.item).fetch1("time_scale")

        # resampling period
        period = (Rate & key).link.period

        # resampling flip index
        index = flip_index(video.times * time_scale, period)

        # resampled and resized video
        return video.array[index]

    @rowmethod
    def visual_trial_ids(self, video_id, trial_filterset_id):
        from foundation.recording.compute_visual import VisualTrials

        # visual trialset key
        key = {
            "trialset_id": (recording.ScanRecording & self.item).fetch1("trialset_id"),
            "trial_filterset_id": trial_filterset_id,
            "video_id": video_id,
        }
        # visual trial_ids
        return (VisualTrials & key).trial_ids

    def _trial_traces(self, trial_ids, traceset_index, key, use_cache=False):
        from foundation.recording import cache, compute_trace as compute

        if use_cache:
            # trial keys
            keys = [{"trial_id": _} for _ in trial_ids]

            # populate cached traces
            with cache_rowproperty():
                cache.ResampledTraces.populate(key, keys, display_progress=True, reserve_jobs=True)

            # load cached traces
            trials = map(np.load, ((cache.ResampledTraces & key & _).fetch1("traces") for _ in keys))

        if traceset_index is None:
            # standardize traces
            transform = (compute.StandardizedTraces & key).transform

            if not use_cache:
                # compute traces
                trials = (compute.ResampledTraces & key).trials(trial_ids=trial_ids)

        else:
            # specific trace
            trace = (recording.TraceSet.Member & key & {"traceset_index": traceset_index}).fetch1()

            # standardize trace
            transform = (compute.StandardizedTrace & trace & key).transform

            if use_cache:
                # index trace
                trials = (_[:, traceset_index] for _ in trials)
            else:
                # compute trace
                trials = (compute.ResampledTrace & trace & key).trials(trial_ids=trial_ids)

        # transformed and resampled trace(s)
        return map(transform, trials)

    @rowmethod
    def trial_perspectives(self, trial_ids, perspective_index=None, use_cache=False):
        # perspective trace(s)
        return self._trial_traces(trial_ids, perspective_index, self.key_perspective, use_cache)

    @rowmethod
    def trial_modulations(self, trial_ids, modulation_index=None, use_cache=False):
        # modulation trace(s)
        return self._trial_traces(trial_ids, modulation_index, self.key_modulation, use_cache)

    @rowmethod
    def trial_units(self, trial_ids, unit_index=None, use_cache=False):
        # unit trace(s)
        return self._trial_traces(trial_ids, unit_index, self.key_unit, use_cache)
