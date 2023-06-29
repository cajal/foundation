import numpy as np
from djutils import keys, merge, rowproperty, cache_rowproperty, MissingError
from foundation.utils import tqdm, logger
from foundation.virtual import utility, stimulus, recording


@keys
class VisualTrials:
    """Visual Trials"""

    @property
    def keys(self):
        return [
            recording.TrialSet,
            recording.TrialFilterSet,
            stimulus.Video,
        ]

    @rowproperty
    def trial_ids(self):
        """
        Returns
        -------
        Tuple[str]
            tuple of keys (foundation.recording.trial.Trial) -- ordered by trial start time
        """
        from foundation.recording.trial import Trial, TrialSet, TrialVideo, TrialBounds, TrialFilterSet

        # all trials
        trials = Trial & (TrialSet & self.key).members

        # filtered trials
        trials = (TrialFilterSet & self.key).filter(trials)

        # video trials
        trials = merge(trials, TrialBounds, TrialVideo) & self.key

        # trial ids, ordered by trial start
        return tuple(trials.fetch("trial_id", order_by="start"))


@keys
class VisualResponse:
    """Visual Response"""

    @property
    def keys(self):
        return [
            recording.Trace,
            recording.TrialFilterSet,
            stimulus.Video,
            utility.Resample,
            utility.Offset,
            utility.Rate,
        ]

    @rowproperty
    def trials(self):
        """
        Returns
        -------
        foundation.utils.response.Trials
            visual response trials -- ordered by trial start time
        """
        from foundation.recording.compute_trace import ResampledTrace
        from foundation.utils.response import Trials

        # visual trials
        key = merge(self.key, recording.TraceTrials)
        trial_ids = (VisualTrials & key).trial_ids

        if trial_ids:
            # trial responses
            responses = (ResampledTrace & self.key).trials(trial_ids)
            return Trials(responses, index=trial_ids, tolerance=1)

        else:
            # no trials
            raise MissingError("No trials found")


@keys
class VisualMeasure:
    """Visual Measure"""

    @property
    def keys(self):
        return [
            recording.Trace,
            recording.TrialFilterSet,
            stimulus.VideoSet,
            utility.Resample,
            utility.Offset,
            utility.Rate,
            utility.Measure,
            utility.Burnin,
        ]

    @rowproperty
    def measure(self):
        """
        Returns
        -------
        float
            visual response measure
        """
        from foundation.stimulus.video import VideoSet
        from foundation.utility.response import Measure
        from foundation.utils.response import concatenate

        # videos
        videos = (VideoSet & self.key).members.fetch("video_id", order_by="video_id", as_dict=True)
        videos = tqdm(videos, desc="Videos")

        # response burnin
        burnin = self.key.fetch1("burnin")

        # trial responses
        with cache_rowproperty():
            responses = [(VisualResponse & self.key & video).trials for video in videos]
            responses = concatenate(*responses, burnin=burnin)

        # response measure
        return (Measure & self.key).link.measure(responses)
