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
        trials = Trial & (TrialSet & self.item).members

        # filtered trials
        trials = (TrialFilterSet & self.item).filter(trials)

        # video trials
        trials = merge(trials, TrialBounds, TrialVideo) & self.item

        # trial ids, ordered by trial start
        return tuple(trials.fetch("trial_id", order_by="start"))


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
        from foundation.recording.compute.resample import ResampledTrace
        from foundation.stimulus.video import VideoSet
        from foundation.utility.response import Measure
        from foundation.utils.response import Trials, concatenate

        # trial set
        trialset = (recording.TraceTrials & self.item).fetch1()

        # videos
        videos = (VideoSet & self.item).members
        videos = videos.fetch("KEY", order_by=videos.primary_key)

        with cache_rowproperty():

            # visual responses
            responses = []

            for video in tqdm(videos, desc="Videos"):

                # trial ids
                trial_ids = (VisualTrials & trialset & video & self.item).trial_ids

                # no trials for video
                if not trial_ids:
                    logger.warning(f"No trials found for video_id `{video['video_id']}`")
                    continue

                # trial responses
                trials = (ResampledTrace & self.item).trials(trial_ids=trial_ids)
                trials = Trials(trials, index=trial_ids, tolerance=1)

                # append
                responses.append(trials)

        # no trials at all
        if not responses:
            logger.warning(f"No trials found")
            return

        # concatenated responses
        responses = concatenate(*responses, burnin=self.item["burnin"])

        # response measure
        return (Measure & self.item).link.measure(responses)
