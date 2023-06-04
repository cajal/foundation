import numpy as np
from djutils import keys, merge, rowmethod, rowproperty, cache_rowproperty, RestrictionError
from foundation.utils import tqdm, disable_tqdm
from foundation.virtual import utility, stimulus, recording, fnn


# ----------------------------- Visual Input -----------------------------

# -- Visual Input Base --


class VisualInput:
    """Visual Input"""

    @rowmethod
    def stimuli(self, video_id, trial_filterset_id=None):
        """
        Parameters
        ----------
        video_id : str
            key -- foundation.stimulus.video.Video
        trial_filterset_id : str | None
            key -- foundation.recording.trial.TrialFilterSet | None

        Yields
        ------
        4D array -- [trials, height, width, channels]
            video frame
        """
        raise NotImplementedError()

    @rowmethod
    def perspectives(self, video_id, trial_filterset_id=None):
        """
        Parameters
        ----------
        video_id : str
            key -- foundation.stimulus.video.Video
        trial_filterset_id : str | None
            key -- foundation.recording.trial.TrialFilterSet | None

        Yields
        ------
        2d array -- [trials, perspectives]
            perspective frame
        """
        raise NotImplementedError()

    @rowmethod
    def modulations(self, video_id, trial_filterset_id=None):
        """
        Parameters
        ----------
        video_id : str
            key -- foundation.stimulus.video.Video
        trial_filterset_id : str | None
            key -- foundation.recording.trial.TrialFilterSet | None

        Yields
        ------
        2d array -- [trials, modulations]
            modulation frame
        """
        raise NotImplementedError()


# -- Visual Input Types --


@keys
class VisualScanTrials:
    """Visual Scan Recording Trials"""

    @property
    def key_list(self):
        return [
            fnn.VisualScan.proj(fnn_filterset_id="trial_filterset_id"),
            stimulus.Video,
            recording.TrialFilterSet,
        ]

    @rowproperty
    def trial_ids(self):
        from foundation.recording.trial import Trial, TrialSet, TrialFilterSet

        # all trials
        key = recording.ScanRecording & self.key
        trials = Trial & (TrialSet & key).members

        # filtered trials
        trials = (TrialFilterSet & self.key).filter(trials)

        # video trials
        trials = merge(trials, recording.TrialVideo) & self.key

        # trial ids, sorted
        return trials.fetch("trial_id", order_by="trial_id").tolist()


@keys
class VisualScan:
    """Visual Scan Recording Trials"""

    @property
    def key_list(self):
        return [fnn.VisualScan]

    @rowmethod
    def trial_ids(self, video_id, trial_filterset_id=None):
        from foundation.recording.trial import Trial, TrialSet, TrialFilterSet

        if trial_filterset_id is None:
            return []

        # all trials
        key = recording.ScanRecording & self.key
        trials = Trial & (TrialSet & key).members

        # filtered trials
        trials = (TrialFilterSet & {"trial_filterset_id": trial_filterset_id}).filter(trials)

        # video trials
        trials = merge(trials, recording.TrialVideo) & {"video_id": video_id}

        # trial ids, sorted
        return trials.fetch("trial_id", order_by="trial_id").tolist()

    @rowmethod
    def stimuli(self, video_id, trial_filterset_id=None):
        from foundation.stimulus.compute_video import ResizedVideo
        from foundation.recording.compute_trial import ResampledTrial
        from foundation.utility.resample import Rate
        from foundation.utils.resample import flip_index, truncate

        # video
        video = (ResizedVideo & {"video_id": video_id}).video
        varray = video.array

        # trials
        trial_ids = self.trial_ids(video_id, trial_filterset_id)

        if trial_ids:
            trial_ids = tqdm(trial_ids, desc="Stimuli")

            # video indexes
            indexes = []
            with cache_rowproperty(), disable_tqdm():
                for trial_id in trial_ids:
                    key = {"trial_id": trial_id}
                    index = (ResampledTrial & key & self.key).flip_index
                    indexes.append(index)

            # truncate and squeeze indexes
            indexes = truncate(*indexes)
            indexes = np.stack(indexes, axis=1)
            if not np.diff(indexes, axis=1).any():
                indexes = indexes[:, :1]

        else:
            # time scale
            time_scale = merge(self.key, recording.ScanVideoTimeScale).fetch1("time_scale")

            # sampling rate
            period = (Rate & self.key).link.period

            # video index
            indexes = flip_index(video.times * time_scale, period)[:, None]

        # yield video frames
        for i in indexes:
            yield varray[i]

    @rowmethod
    def perspectives(self, video_id, trial_filterset_id=None):
        from foundation.fnn.compute_dataset import VisualScan
        from foundation.recording.compute_trace import StandardTraces, ResampledTraces
        from foundation.utils.resample import truncate

        # traceset key
        key = (VisualScan & self.key).perspectives_key

        # traceset transform
        transform = (StandardTraces & key).transform

        # trials
        trial_ids = self.trial_ids(video_id, trial_filterset_id)

        if trial_ids:
            trial_ids = tqdm(trial_ids, desc="Perspectives")
        else:
            return

        # resampled traceset
        trials = []
        with cache_rowproperty(), disable_tqdm():
            for trial_id in trial_ids:
                trial = (ResampledTraces & {"trial_id": trial_id} & key).trial
                trial = transform(trial)
                trials.append(trial)

        # stacked traceset
        traces = truncate(*trials)
        traces = np.stack(traces, axis=1)

        # yield traceset frames
        def frames():
            yield from traces

        return frames()

    @rowmethod
    def modulations(self, video_id, trial_filterset_id=None):
        from foundation.fnn.compute_dataset import VisualScan
        from foundation.recording.compute_trace import StandardTraces, ResampledTraces
        from foundation.utils.resample import truncate

        # traceset key
        key = (VisualScan & self.key).modulations_key

        # traceset transform
        transform = (StandardTraces & key).transform

        # trials
        trial_ids = self.trial_ids(video_id, trial_filterset_id)

        if trial_ids:
            trial_ids = tqdm(trial_ids, desc="Modulations")
        else:
            return

        # resampled traceset
        trials = []
        with cache_rowproperty(), disable_tqdm():
            for trial_id in trial_ids:
                trial = (ResampledTraces & {"trial_id": trial_id} & key).trial
                trial = transform(trial)
                trials.append(trial)

        # stacked traceset
        traces = truncate(*trials)
        traces = np.stack(traces, axis=1)

        # yield traceset frames
        def frames():
            yield from traces

        return frames()
