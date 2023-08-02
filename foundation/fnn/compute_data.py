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
            number of stimulus channels
        """
        raise NotImplementedError()

    @rowproperty
    def perspectives(self):
        """
        Returns
        -------
        int
            number of perspective features
        """
        raise NotImplementedError()

    @rowproperty
    def modulations(self):
        """
        Returns
        -------
        int
            number of modulations features
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
    def perspective_offset(self):
        """
        Returns
        -------
        float
            perspective sampling offset (seconds)
        """
        raise NotImplementedError()

    @rowproperty
    def modulation_offset(self):
        """
        Returns
        -------
        float
            modulation sampling offset (seconds)
        """
        raise NotImplementedError()

    @rowproperty
    def unit_offset(self):
        """
        Returns
        -------
        float
            unit sampling offset (seconds)
        """
        raise NotImplementedError()

    @rowproperty
    def sampling_period(self):
        """
        Returns
        -------
        float
            sampling period (seconds)
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


class VisualType(DataType):
    """Visual Data"""

    @rowproperty
    def resolution(self):
        """
        Returns
        -------
        int
            height (pixels)
        int
            width (pixels)
        """
        raise NotImplementedError()

    @rowproperty
    def resize_id(self):
        """
        Returns
        -------
        str
            key (foundation.utility.resize.Resize)
        """
        raise NotImplementedError()


class RecordingType(DataType):
    """Recording Data"""

    @rowmethod
    def trial_perspectives(self, trial_ids):
        """
        Parameters
        ----------
        trial_id : Sequence[str]
            sequence of keys (foundation.recording.trial.Trial)

        Yields
        ------
        2D array
            [samples, perspectives]
        """
        raise NotImplementedError()

    @rowmethod
    def trial_modulations(self, trial_ids):
        """
        Parameters
        ----------
        trial_id : Sequence[str]
            sequence of keys (foundation.recording.trial.Trial)

        Yields
        ------
        2D array
            [samples, modulations]
        """
        raise NotImplementedError()

    @rowmethod
    def trial_units(self, trial_ids):
        """
        Parameters
        ----------
        trial_id : Sequence[str]
            sequence of keys (foundation.recording.trial.Trial)

        Yields
        ------
        2D array
            [samples, units]
        """
        raise NotImplementedError()


class VisualRecordingType(VisualType, RecordingType):
    """Visual Recording"""

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
    def visual_trial_stimulus(self, video_id):
        """
        Parameters
        ----------
        video_id : str
            key (foundation.stimulus.video.Video)

        Returns
        -------
        4D array
            [samples, height, width, channels] (uint8) -- video frames
        """
        raise NotImplementedError()


# -- Data Types --


@keys
class VisualScan(VisualRecordingType):
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
        return (recording.ScanTrials * fnn.Spec.VisualSpec).proj("rate_id", "trialset_id", **keymap)

    @rowproperty
    def key_perspective(self):
        return (self._key_traces("perspective") * recording.ScanVisualPerspectives & self.item).fetch1()

    @rowproperty
    def key_modulation(self):
        return (self._key_traces("modulation") * recording.ScanVisualModulations & self.item).fetch1()

    @rowproperty
    def key_unit(self):
        return (self._key_traces("unit") * recording.ScanUnits & self.item).fetch1()

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
    def perspective_offset(self):
        from foundation.utility.resample import Offset

        return (Offset & self.key_perspective).link.offset

    @rowproperty
    def modulation_offset(self):
        from foundation.utility.resample import Offset

        return (Offset & self.key_modulation).link.offset

    @rowproperty
    def unit_offset(self):
        from foundation.utility.resample import Offset

        return (Offset & self.key_unit).link.offset

    @rowproperty
    def sampling_period(self):
        from foundation.utility.resample import Rate

        return (Rate & self.key_video).link.period

    @rowproperty
    def resolution(self):
        key = self.key_video
        return key["height"], key["width"]

    @rowproperty
    def resize_id(self):
        return self.key_video["resize_id"]

    @rowproperty
    def dataset(self):
        from fnn.data import NpyFile, Dataset
        from foundation.recording.trace import TraceSet
        from foundation.recording.compute_trace import StandardizedTraces
        from foundation.recording.scan import ScanUnitOrder, ScanVisualPerspectiveOrder, ScanVisualModulationOrder

        # keys
        key_v = self.key_video
        key_p = self.key_perspective
        key_m = self.key_modulation
        key_u = self.key_unit

        # transforms
        transform_p = (StandardizedTraces & key_p).transform
        transform_m = (StandardizedTraces & key_m).transform
        transform_u = (StandardizedTraces & key_u).transform

        # traces
        traces_p = merge((TraceSet & key_p).members, ScanVisualPerspectiveOrder & key_p)
        traces_m = merge((TraceSet & key_m).members, ScanVisualModulationOrder & key_m)
        traces_u = merge((TraceSet & key_u).members, ScanUnitOrder & key_u)

        # orders
        order_p = traces_p.fetch("traceset_index", order_by="trace_order")
        order_m = traces_m.fetch("traceset_index", order_by="trace_order")
        order_u = traces_u.fetch("traceset_index", order_by="trace_order")

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
            trial_stimuli = video[index].astype(np.uint8)

            # perspectives
            traces = (recording.ResampledTraces & trial & key_p).fetch1("traces")
            trial_perspectives = transform_p(traces).astype(np.float32)[:, order_p]

            # modulations
            traces = (recording.ResampledTraces & trial & key_m).fetch1("traces")
            trial_modulations = transform_m(traces).astype(np.float32)[:, order_m]

            # units
            traces = (recording.ResampledTraces & trial & key_u).fetch1("traces")
            trial_units = transform_u(traces).astype(np.float32)[:, order_u]

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

    def _trial_traces(self, trial_ids, datatype):
        from foundation.recording.trace import TraceSet
        from foundation.recording.compute_trace import StandardizedTraces
        from foundation.recording.scan import ScanVisualPerspectiveOrder, ScanVisualModulationOrder, ScanUnitOrder

        if datatype == "perspective":
            key = self.key_perspective
            order = ScanVisualPerspectiveOrder

        elif datatype == "modulation":
            key = self.key_modulation
            order = ScanVisualModulationOrder

        elif datatype == "unit":
            key = self.key_unit
            order = ScanUnitOrder

        else:
            raise ValueError(f"datatype `{datatype}` not recognized")

        # trace standardization
        transform = (StandardizedTraces & key).transform

        # trace order
        order = merge((TraceSet & key).members, order & key)
        order = order.fetch("traceset_index", order_by="trace_order")

        # load trials
        for trial_id in trial_ids:

            traces = (recording.ResampledTraces & key & {"trial_id": trial_id}).fetch1("traces")
            yield transform(traces).astype(np.float32)[:, order]

    @rowmethod
    def trial_perspectives(self, trial_ids):
        # perspective trace(s)
        return self._trial_traces(trial_ids, "perspective")

    @rowmethod
    def trial_modulations(self, trial_ids):
        # modulation trace(s)
        return self._trial_traces(trial_ids, "modulation")

    @rowmethod
    def trial_units(self, trial_ids):
        # unit trace(s)
        return self._trial_traces(trial_ids, "unit")

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

    @rowmethod
    def visual_trial_stimulus(self, video_id):
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
