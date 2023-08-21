import numpy as np
import pandas as pd
from djutils import keys, merge, unique, rowproperty, rowmethod
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

    @rowproperty
    def trialset_id(self):
        """
        Returns
        -------
        str
            key (foundation.recording.trial.TrialSet)
        """
        raise NotImplementedError()

    @rowmethod
    def trial_stimuli(self, trial_ids):
        """
        Parameters
        ----------
        trial_id : Sequence[str]
            sequence of keys (foundation.recording.trial.Trial)

        Yields
        ------
        ND array
            [samples, ...]
        """
        raise NotImplementedError()

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


# -- Data Types --


class _VisualScan(VisualType, RecordingType):
    """Visual Scan Data -- Base"""

    @property
    def unit_set(self):
        raise NotImplementedError()

    @property
    def unit_order(self):
        raise NotImplementedError()

    @rowproperty
    def key_video(self):
        return (fnn.Spec.VisualSpec & self.item).proj("height", "width", "resize_id", "rate_id").fetch1()

    def _key_traces(self, datatype):
        keymap = {f"{_}_id": f"{_}_id_{datatype}" for _ in ["resample", "offset", "standardize"]}
        return (recording.ScanTrials * fnn.Spec.VisualSpec).proj("rate_id", "trialset_id", **keymap)

    @rowproperty
    def key_perspective(self):
        return (self._key_traces("perspective") * recording.ScanVisualPerspectives & self.item).fetch1()

    @rowproperty
    def key_modulation(self):
        return (self._key_traces("modulation") * recording.ScanVisualModulations & self.item).fetch1()

    @rowproperty
    def key_unit(self):
        return (self._key_traces("unit") * self.unit_set & self.item).fetch1()

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
    def trialset_id(self):
        return (recording.ScanRecording & self.item).fetch1("trialset_id")

    @rowmethod
    def trial_stimuli(self, trial_ids):
        key = self.key_video
        rows = stimulus.ResizedVideo * recording.TrialVideo * recording.ResampledTrial

        # load trials
        for trial_id in trial_ids:
            video, index = (rows & key & {"trial_id": trial_id}).fetch1("video", "index")
            yield video[index].astype(np.uint8)

    def _trial_traces(self, trial_ids, datatype):
        from foundation.recording.trace import TraceSet
        from foundation.recording.compute.standardize import StandardizedTraces

        if datatype == "perspective":
            key = self.key_perspective
            order = recording.ScanVisualPerspectiveOrder

        elif datatype == "modulation":
            key = self.key_modulation
            order = recording.ScanVisualModulationOrder

        elif datatype == "unit":
            key = self.key_unit
            order = self.unit_order

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

    @rowproperty
    def dataset(self):
        from fnn.data import NpyFile, Dataset

        # tiers
        training_tier, validation_tier = self.key.fetch1("training_tier", "validation_tier")
        tier_keys = [{"tier_index": index} for index in [training_tier, validation_tier]]

        # trials
        trials = merge(
            recording.ScanTrials & self.item,
            recording.TrialTier & self.item & tier_keys,
            recording.TrialBounds,
            recording.TrialSamples,
        )
        trial_ids, tiers, samples = trials.fetch("trial_id", "tier_index", "samples", order_by="start")

        # load trials
        stimuli, perspectives, modulations, units = [], [], [], []

        for s, p, m, u in zip(
            self.trial_stimuli(tqdm(trial_ids, desc="Trials")),
            self.trial_perspectives(trial_ids),
            self.trial_modulations(trial_ids),
            self.trial_units(trial_ids),
        ):

            stimuli.append(NpyFile(s))
            perspectives.append(NpyFile(p))
            modulations.append(NpyFile(m))
            units.append(NpyFile(u))

        assert len(stimuli) == len(trial_ids)

        # dataset
        data = {
            "training": tiers == training_tier,
            "samples": samples,
            "stimuli": stimuli,
            "perspectives": perspectives,
            "modulations": modulations,
            "units": units,
        }
        data = pd.DataFrame(data, index=pd.Index(trial_ids, name="trial_id"))
        return Dataset(data)


@keys
class VisualScan(_VisualScan):
    """Visual Scan Data -- Activity"""

    @property
    def keys(self):
        return [
            fnn.VisualScan,
        ]

    @property
    def unit_set(self):
        return recording.ScanUnits

    @property
    def unit_order(self):
        return recording.ScanUnitOrder


@keys
class VisualScanRaw(_VisualScan):
    """Visual Scan Data -- Fluorescence"""

    @property
    def keys(self):
        return [
            fnn.VisualScanRaw,
        ]

    @property
    def unit_set(self):
        return recording.ScanUnitsRaw

    @property
    def unit_order(self):
        return recording.ScanUnitRawOrder


@keys
class Sensorium2023(DataType):
    """Sensorium 2023"""

    @property
    def keys(self):
        return [
            fnn.Sensorium2023,
        ]

    @rowproperty
    def root(self):
        from os.path import join

        return join("/mnt", "scratch09", "sensorium_2023", self.item["sensorium_dataset"])

    def trim(self, x, axis):
        end = np.where(np.isnan(x))[axis].min()
        return np.take(x, np.arange(end), axis=axis)
