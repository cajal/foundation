import numpy as np
from djutils import keys, rowproperty
from foundation.utils import tqdm
from foundation.virtual import utility, stimulus, recording, fnn


@keys
class NetworkModelRecording:
    """Network Model Recording"""

    @property
    def key_list(self):
        return [
            stimulus.Video,  # visual stimulus
            fnn.NetworkModel,  # network model
            recording.TrialFilterSet,  # trial filter
            utility.Bool.proj(trial_perspective="bool"),  # trial | default perspective
            utility.Bool.proj(trial_modulation="bool"),  # trial | default modulation
        ]

    @rowproperty
    def response(self):
        """
        Returns
        -------
        3D array
            [samples, trials, units] -- dtype=float-like
        List[str] | None
            list of trial_ids -- key (foundation.recording.trial.Trial)
        """
        from foundation.fnn.network import Network
        from foundation.fnn.model import NetworkModel

        # load model
        model = (NetworkModel & self.key).model

        # input arguments
        video_id, trial_perspective, trial_modulation, trial_filterset_id = self.key.fetch1(
            "video_id", "trial_perspective", "trial_modulation", "trial_filterset_id"
        )

        # visual inputs, trial_ids
        stimuli, perspectives, modulations, trial_ids = (Network & self.key).link.compute_data.visual_inputs(
            video_id=video_id,
            trial_perspective=trial_perspective,
            trial_modulation=trial_modulation,
            trial_filterset_id=trial_filterset_id,
        )
        assert trial_ids, "No trials found"

        # visual responses
        responses = model.generate_response(
            stimuli=stimuli,
            perspectives=perspectives,
            modulations=modulations,
        )
        responses = tqdm(responses, desc="Responses")
        responses = np.stack(list(responses), axis=0)

        # responses and trial_ids
        return responses, trial_ids
