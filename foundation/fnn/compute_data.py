import numpy as np
import pandas as pd
from datajoint import U
from djutils import keys, merge, rowproperty, rowmethod, cache_rowproperty
from foundation.utils import logger, tqdm
from foundation.virtual import utility, stimulus, recording, fnn


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
    def trainset(self):
        """
        Returns
        -------
        fnn.data.Dataset
            training dataset
        """
        raise NotImplementedError()

    @rowmethod
    def trials(self, training=None):
        """
        Parameters
        ----------
        training : bool | None
            True (training trials) | False (non-training trials) | None (all trials)

        Returns
        -------
        foundation.recording.trial.Trial (rows)
            recording trials
        """
        raise NotImplementedError()

    @rowmethod
    def visual_inputs(self, video_id, perspectives=True, modulations=True, training=None):
        """
        Parameters
        ----------
        video_id : str
            key (foundation.stimulus.video.Video)
        perspectives : bool
            True (return trial perspectives) | False (return None)
        modulations : bool
            True (return trial modulations) | False (return None)
        training : bool | None
            True (training trials) | False (non-training trials) | None (all trials)

        Returns
        -------
        Iterable[3D array]
            video frames -- [height, width, channels] x samples --- dtype=uint8
        None | Iterable[2D array]
            trial perspectives -- [trials, perspectives] x samples -- dtype=float-like
        None | Iterable[2D array]
            trial modulations -- [trials, modulations] x samples -- dtype=float-like
        None | List[str]
            list of trial_ids -- key (foundation.recording.trial.Trial)
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
    def training_stimuli(self):
        return merge(
            self.key.proj(spec_id="stimuli_id"),
            fnn.Spec.VideoSpec,
        ).fetch1()

    @rowproperty
    def training_perspectives(self):
        return merge(
            self.key.proj(spec_id="perspectives_id"),
            fnn.Spec.TraceSpec,
            recording.ScanVisualPerspectives,
            recording.ScanTrials,
        ).fetch1()

    @rowproperty
    def training_modulations(self):
        return merge(
            self.key.proj(spec_id="modulations_id"),
            fnn.Spec.TraceSpec,
            recording.ScanVisualModulations,
            recording.ScanTrials,
        ).fetch1()

    @rowproperty
    def training_units(self):
        return merge(
            self.key.proj(spec_id="units_id"),
            fnn.Spec.TraceSpec,
            recording.ScanUnits,
            recording.ScanTrials,
        ).fetch1()

    @rowproperty
    def sizes(self):
        # number of stimulus channels
        stimuli = merge(self.trials(training=True), recording.TrialVideo, stimulus.VideoInfo)
        stimuli = (U("channels") & stimuli).fetch1("channels")
        sizes = [stimuli]

        # number of perspective, modulation, and unit traces
        for attr in ["perspectives", "modulations", "units"]:
            traces = recording.TraceSet & getattr(self, f"training_{attr}")
            traces = traces.fetch1("members")
            sizes.append(traces)

        return tuple(sizes)

    @rowproperty
    def timing(self):
        from foundation.utility.resample import Rate, Offset

        # sampling period
        period = (Rate & self.key).link.period

        # response offset
        offset = (Offset & self.key).link.offset

        return period, offset

    @rowmethod
    def trials(self, training=None):
        from foundation.recording.trial import Trial, TrialSet

        if training is None:
            # all scan trials
            key = merge(self.key, recording.ScanRecording)
            trials = Trial & (TrialSet & key).members

        elif training:
            # training trials
            key = merge(self.key, recording.ScanTrials)
            trials = Trial & (TrialSet & key).members

        else:
            # non-training trials
            key = merge(self.key, recording.ScanRecording)
            trials = Trial & (TrialSet & key).members

            key = merge(self.key, recording.ScanTrials)
            trials = trials - (TrialSet & key).members

        return trials

    @rowproperty
    def trainset(self):
        from foundation.recording.compute_trace import StandardizedTraces
        from fnn.data import NpyFile, Dataset

        # data keys
        key_s = self.training_stimuli
        key_p = self.training_perspectives
        key_m = self.training_modulations
        key_u = self.training_units

        # data transforms
        transform_p = (StandardizedTraces & key_p).transform
        transform_m = (StandardizedTraces & key_m).transform
        transform_u = (StandardizedTraces & key_u).transform

        # data lists
        stimuli, perspectives, modulations, units = [], [], [], []

        # training trials
        trials = merge(
            self.trials(training=True),
            recording.TrialBounds,
            recording.TrialSamples,
            recording.TrialVideo,
            stimulus.Video,
        )
        trials, index, samples = trials.fetch("KEY", "trial_id", "samples", order_by="start")

        # load trials
        for trial in tqdm(trials, desc="Trials"):

            # stimuli
            video, imap = (stimulus.ResizedVideo * recording.ResampledTrial & trial & key_s).fetch1("video", "index")
            trial_stimuli = NpyFile(video, indexmap=np.load(imap), dtype=np.uint8)

            # perspectives
            traces = (recording.ResampledTraces & trial & key_p).fetch1("traces")
            trial_perspectives = NpyFile(traces, transform=transform_p, dtype=np.float32)

            # modulations
            traces = (recording.ResampledTraces & trial & key_m).fetch1("traces")
            trial_modulations = NpyFile(traces, transform=transform_m, dtype=np.float32)

            # units
            traces = (recording.ResampledTraces & trial & key_u).fetch1("traces")
            trial_units = NpyFile(traces, transform=transform_u, dtype=np.float32)

            # append
            stimuli.append(trial_stimuli)
            perspectives.append(trial_perspectives)
            modulations.append(trial_modulations)
            units.append(trial_units)

        # training dataset
        data = {
            "samples": samples,
            "stimuli": stimuli,
            "perspectives": perspectives,
            "modulations": modulations,
            "units": units,
        }
        return Dataset(data, index=pd.Index(index, name="trial_id"))
