import numpy as np
from djutils import keys, merge, rowproperty, cache_rowproperty
from foundation.utils import tqdm
from foundation.virtual import utility, stimulus, recording


@keys
class Trace:
    """Visual Trace"""

    @property
    def key_list(self):
        return [
            stimulus.Video,  # visual stimulus
            recording.Trace,  # recording trace
            recording.TrialFilterSet,  # recording trial filter
            recording.TrialSet,  # standardization trial set
            utility.Standardize,  # standardization method
            utility.Resample,  # resampling method
            utility.Offset,  # resampling offset
            utility.Rate,  # resampling rate
        ]

    @rowproperty
    def responses(self):
        """
        Returns
        -------
        2D array
            [samples, trials] -- dtype=float-like
        List[str]
            list of trial_ids -- key (foundation.recording.trial.Trial), ordered by trial start
        """
        from foundation.recording.trial import Trial, TrialSet, TrialFilterSet, TrialVideo, TrialBounds
        from foundation.recording.compute_trace import ResampledTrace, StandardizedTrace
        from foundation.utils.resample import truncate

        # all trials
        key = recording.Trace & {"trace_id": self.key.fetch1("trace_id")}
        trials = merge(key, recording.TraceTrials)
        trials = Trial & (TrialSet & trials).members

        # filtered trials
        trials = (TrialFilterSet & self.key).filter(trials)

        # video trials
        trials = merge(trials, TrialVideo) & {"video_id": self.key.fetch1("video_id")}

        # trial ids, ordered by trial start
        trials = merge(trials, TrialBounds).fetch("trial_id", order_by="start").tolist()
        assert trials, "No trials found"

        # trial responses
        responses = (ResampledTrace & self.key).trials(trial_ids=trials)
        responses = tqdm(responses, desc="Responses")

        # response tranform
        transform = (StandardizedTrace & self.key).transform

        # compute and transform responses
        responses = np.stack(truncate(*map(transform, responses), tolerance=1), axis=1)

        # responses and trial_ids
        return responses, trials
