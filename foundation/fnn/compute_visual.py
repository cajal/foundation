import numpy as np
from djutils import keys, rowproperty
from foundation.utils import tqdm
from foundation.virtual import utility, stimulus, recording, fnn


@keys
class NetworkModelTrial:
    """Network Model Trial"""

    @property
    def key_list(self):
        return [
            stimulus.Video,
            fnn.NetworkModel,
            utility.Bool.proj(trial_perspectives="bool"),
            utility.Bool.proj(trial_modulations="bool"),
            recording.TrialFilterSet,
        ]

    @rowproperty
    def responses(self):
        """
        Returns
        -------
        2D array | 3D array
            [samples, units] | [samples, trials, units] -- dtype=float-like
        None | List[str]
            None | list of trial_ids -- key (foundation.recording.trial.Trial), ordered by trial start
        """
        from foundation.fnn.network import Network
        from foundation.fnn.model import NetworkModel

        # load model
        model = (NetworkModel & self.key).model

        # model inputs
        video_id, trial_perspectives, trial_modulations, trial_filterset_id = self.key.fetch1(
            "video_id", "trial_perspectives", "trial_modulations", "trial_filterset_id"
        )
        stimuli, perspectives, modulations, trial_ids = (Network & self.key).link.data.visual_inputs(
            video_id=video_id,
            trial_perspectives=trial_perspectives,
            trial_modulations=trial_modulations,
            trial_filterset_id=trial_filterset_id,
        )

        # generate responses
        responses = model.generate_responses(
            stimuli=stimuli,
            perspectives=perspectives,
            modulations=modulations,
        )
        responses = tqdm(responses, desc="Responses")
        responses = np.stack(list(responses), axis=0)

        # responses and trial_ids
        return responses, trial_ids
