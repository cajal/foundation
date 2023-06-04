import numpy as np
from djutils import keys, merge, rowmethod, rowproperty, cache_rowproperty
from foundation.utils import tqdm, disable_tqdm
from foundation.virtual import utility, stimulus, recording, fnn


# ----------------------------- Visual Input -----------------------------

# -- Visual Input Base --


class Visual:
    """Visual Input"""

    @rowmethod
    def inputs(self, video_id, trial_filterset_id=None):
        """
        Parameters
        ----------
        video_id : str
            key -- foundation.stimulus.video.Video
        trial_filterset_id : str | None
            key -- foundation.recording.trial.TrialFilterSet | None

        Returns
        -------
        Iterable[4D array]
            [trials, height, width, channels] x timepoints --- dtype=unit8
        Iterable[2D array] | None
            [trials, perspectives] x timepoints --- dtype=float | None
        Iterable[2D array] | None
            [trials, modulations] x timepoints --- dtype=float | None
        """
        raise NotImplementedError()


# -- Visual Input Types --


@keys
class VisualScan(Visual):
    """Visual Scan Inputs"""

    @property
    def key_list(self):
        return [fnn.VisualScan]

    @rowmethod
    def inputs(self, video_id, trial_filterset_id=None):
        from foundation.utils.resample import flip_index, truncate
        from foundation.utility.resample import Rate
        from foundation.recording.trial import Trial, TrialSet, TrialFilterSet
        from foundation.stimulus.compute_video import ResizedVideo
        from foundation.recording.compute_trial import ResampledTrial
        from foundation.recording.compute_trace import StandardTraces, ResampledTraces
        from foundation.fnn.compute_dataset import VisualScan as Dataset

        if trial_filterset_id is None:
            # no trials
            trial_ids = []
        else:
            # all trials
            key = recording.ScanRecording & self.key
            trials = Trial & (TrialSet & key).members

            # filtered trials
            trials = (TrialFilterSet & {"trial_filterset_id": trial_filterset_id}).filter(trials)

            # video trials
            trials = merge(trials, recording.TrialVideo) & {"video_id": video_id}

            # trial ids, sorted
            trial_ids = trials.fetch("trial_id", order_by="trial_id").tolist()

        # -------------------- stimuli --------------------

        # video
        video = (ResizedVideo & {"video_id": video_id}).video
        varray = video.array

        if trial_ids:
            # video indexes
            indexes = []
            _trial_ids = tqdm(trial_ids, desc="Stimuli")
            with cache_rowproperty(), disable_tqdm():
                for trial_id in _trial_ids:
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
        def stimuli():
            for i in indexes:
                yield varray[i]

        # -------------------- perspectives --------------------

        if trial_ids:
            # traceset key
            key = (Dataset & self.key).perspectives_key

            # traceset transform
            transform = (StandardTraces & key).transform

            # resampled traceset
            trials = []
            _trial_ids = tqdm(trial_ids, desc="Perspectives")
            with cache_rowproperty(), disable_tqdm():
                for trial_id in _trial_ids:
                    trial = (ResampledTraces & {"trial_id": trial_id} & key).trial
                    trial = transform(trial)
                    trials.append(trial)

            # stacked traceset
            ptraces = truncate(*trials)
            ptraces = np.stack(ptraces, axis=1)

            # yield traceset frames
            def perspectives():
                yield from ptraces

        else:

            def perspectives():
                return

        # -------------------- modulations --------------------

        if trial_ids:
            # traceset key
            key = (Dataset & self.key).modulations_key

            # traceset transform
            transform = (StandardTraces & key).transform

            # resampled traceset
            trials = []
            _trial_ids = tqdm(trial_ids, desc="Modulations")
            with cache_rowproperty(), disable_tqdm():
                for trial_id in _trial_ids:
                    trial = (ResampledTraces & {"trial_id": trial_id} & key).trial
                    trial = transform(trial)
                    trials.append(trial)

            # stacked traceset
            mtraces = truncate(*trials)
            mtraces = np.stack(mtraces, axis=1)

            # yield traceset frames
            def modulations():
                yield from mtraces

        else:

            def modulations():
                return

        return stimuli(), perspectives(), modulations()
