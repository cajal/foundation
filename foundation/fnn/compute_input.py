import numpy as np
from djutils import keys, merge, rowmethod, rowproperty, cache_rowproperty, RestrictionError
from foundation.utils import tqdm, disable_tqdm
from foundation.virtual import utility, stimulus, recording, fnn


# ----------------------------- Visual Input -----------------------------

# -- Visual Input Base --


@keys
class VisualInput:
    """Visual Input"""

    @property
    def key_list(self):
        return [stimulus.Video, fnn.VideoSpec, utility.Rate] + self.input_list

    @property
    def input_list(self):
        raise NotImplementedError()

    @rowproperty
    def time_scale(self):
        """
        Returns
        -------
        float
            video time scale
        """
        raise NotImplementedError()

    @rowmethod
    def stimuli(self):
        """
        Yields
        ------
        4D array -- [trials, height, width, channels]
            video frame
        """
        from foundation.stimulus.compute_video import ResizedVideo
        from foundation.utility.resample import Rate
        from foundation.utils.resample import flip_index

        # video
        video = (ResizedVideo & self.key).video

        # sampling rate
        period = (Rate & self.key).link.period

        # video index
        index = flip_index(video.times * self.time_scale, period)

        # yield video frames
        varray = video.array
        for i in index:
            yield varray[i, None]

    @rowmethod
    def perspectives(self):
        """
        Yields
        ------
        2d array -- [trials, perspectives]
            perspective frame
        """
        return

    @rowmethod
    def modulations(self):
        """
        Yields
        ------
        2d array -- [trials, modulations]
            modulation frame
        """
        return


# -- Visual Input Types --


class VisualScan(VisualInput):
    """Visual Scan Input"""

    @property
    def input_list(self):
        return [
            fnn.VisualScan,
            (fnn.Spec.VideoSpec * fnn.VideoSpec).proj(stimuli_id="spec_id"),
        ]

    @rowproperty
    def time_scale(self):
        return merge(self.key, recording.ScanVideoTimeScale).fetch1("time_scale")


# ----------------------------- Visual Recording Input -----------------------------

# -- Visual Recording Input Base --


class VisualRecordingInput(VisualInput):
    """Visual Recording Input"""

    @property
    def key_list(self):
        return [stimulus.Video, fnn.VideoSpec, utility.Rate, recording.TrialFilterSet] + self.input_list

    @rowproperty
    def all_trials(self):
        """
        Returns
        -------
        foundation.recording.Trial
            tuple(s), all trials
        """
        raise NotImplementedError()

    @rowproperty
    def trial_ids(self):
        """
        Returns
        -------
        list[str]
            list of trial_id's (foundation.recording.trial.Trial)
        """
        from foundation.recording.trial import Trial, TrialBounds, TrialFilterSet

        # all trials
        trials = self.all_trials

        # filtered trials
        trials = (TrialFilterSet & self.key).filter(trials)

        # filtered trials restricted by video
        trials = merge(trials, recording.TrialVideo) & self.key

        if trials:
            return merge(trials, TrialBounds).fetch("trial_id", order_by="start").tolist()
        else:
            raise RestrictionError("No trials found.")

    @rowmethod
    def stimuli(self):
        from foundation.stimulus.compute import ResizedVideo
        from foundation.recording.compute_trial import ResampledTrial
        from foundation.utils.resample import truncate

        # video
        video = (ResizedVideo & self.key).video

        # video indexes
        indexes = []
        trial_ids = tqdm(self.trial_ids, desc="Stimuli")

        with cache_rowproperty(), disable_tqdm():
            for trial_id in trial_ids:
                key = {"trial_id": trial_id}
                index = (ResampledTrial & key & self.key).flip_index
                indexes.append(index)

        indexes = truncate(*indexes)
        indexes = np.stack(indexes, axis=1)
        if not np.diff(indexes, axis=1).any():
            indexes = indexes[:, :1]

        # yield video frames
        varray = video.array
        for i in indexes:
            yield varray[i]


# -- Visual Recording Input Types --


class VisualScanRecording(VisualRecordingInput):
    """Visual Scan Recording Input"""

    @property
    def input_list(self):
        return [
            fnn.VisualScan.proj(fnn_filterset_id="trial_filterset_id"),
            (fnn.Spec.VideoSpec * fnn.VideoSpec).proj(stimuli_id="spec_id"),
        ]

    @rowproperty
    def all_trials(self):
        from foundation.recording.trial import Trial, TrialSet

        trials = TrialSet & merge(self.key, recording.ScanRecording)
        return Trial & trials.members

    @rowmethod
    def _traces(self, input_type):
        from foundation.fnn.compute_dataset import VisualScan
        from foundation.recording.compute_trace import StandardTraces, ResampledTraces
        from foundation.utils.resample import truncate

        if input_type not in ["perspective", "modulation"]:
            raise ValueError("input_type must be either `perspective` or `modulation`")

        # traceset key
        key = VisualScan & self.key.proj(
            input_filterset_id="trial_filterset_id",
            trial_filterset_id="fnn_filterset_id",
        )
        key = getattr(key, f"{input_type}s_key")

        # traceset transform
        transform = (StandardTraces & key).transform

        # trials
        trials = []
        trial_ids = tqdm(self.trial_ids, desc=f"{input_type.capitalize()}s")
        with cache_rowproperty(), disable_tqdm():
            for trial_id in trial_ids:
                trial = (ResampledTraces & {"trial_id": trial_id} & key).trial
                trial = transform(trial)
                trials.append(trial)

        # traceset
        traces = truncate(*trials)
        traces = np.stack(traces, axis=1)

        # yield traceset frames
        def frames():
            yield from traces

        return frames()

    @rowmethod
    def perspectives(self):
        return self._traces("perspective")

    @rowmethod
    def modulations(self):
        return self._traces("modulation")
