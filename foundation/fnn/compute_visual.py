import numpy as np
from djutils import keys, rowproperty, cache_rowproperty, MissingError
from foundation.utils import tqdm
from foundation.virtual import utility, stimulus, recording, fnn


@keys
class VisualNetworkRecording:
    """Network Model Recording"""

    @property
    def keys(self):
        return [
            fnn.NetworkModel,  # network model
            recording.TrialFilterSet,  # trial filter
            utility.Bool.proj(trial_perspective="bool"),  # trial | default perspective
            utility.Bool.proj(trial_modulation="bool"),  # trial | default modulation
            stimulus.Video,  # visual stimulus
        ]

    @rowproperty
    def response(self):
        """
        Returns
        -------
        3D array
            [samples, trials, units] -- visual response
        List[str] | None
            list of trial_ids -- key (foundation.recording.trial.Trial)
        """
        from foundation.fnn.network import Network
        from foundation.fnn.model import NetworkModel
        from foundation.utils.resample import truncate

        # load data
        data = (Network & self.item).link.compute_data

        # load trials
        trial_ids = data.visual_trial_ids(
            video_id=self.item["video_id"],
            trial_filterset_id=self.item["trial_filterset_id"],
        )

        # raise exception if no trials
        if not trial_ids:
            raise MissingError("No trials found")

        # load stimuli
        stimuli = data.visual_stimuli(video_id=self.item["video_id"])

        # load perspectives
        if self.item["trial_perspective"]:
            perspectives = data.trial_perspectives(trial_ids=trial_ids)
            perspectives = np.stack(truncate(*perspectives, tolerance=1), axis=1)
        else:
            perspectives = None

        # load modulations
        if self.item["trial_modulation"]:
            modulations = data.trial_modulations(trial_ids=trial_ids)
            modulations = np.stack(truncate(*modulations, tolerance=1), axis=1)
        else:
            modulations = None

        # load model
        model = (NetworkModel & self.item).model

        # generate responses
        responses = model.generate_response(
            stimuli=stimuli,
            perspectives=perspectives,
            modulations=modulations,
        )
        responses = tqdm(responses, desc="Responses")
        responses = np.stack(list(responses), axis=0)

        if perspectives is None and modulations is None:
            # unsqueeze and repeat trials
            responses = np.expand_dims(responses, axis=1)
            responses = np.repeat(responses, repeats=len(trial_ids), axis=1)

        return responses, trial_ids
