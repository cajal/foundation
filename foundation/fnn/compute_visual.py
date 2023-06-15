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
        Returns (trials exist)
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
        video_id, trial_perspectives, trial_modulations, trial_filterset_id = self.key.fetch1(
            "video_id", "trial_perspectives", "trial_modulations", "trial_filterset_id"
        )

        # visual inputs, trial_ids
        stimuli, perspectives, modulations, trial_ids = (Network & self.key).link.data.visual_inputs(
            video_id=video_id,
            trial_perspectives=trial_perspectives,
            trial_modulations=trial_modulations,
            trial_filterset_id=trial_filterset_id,
        )
        assert trial_ids, "No trials found"

        # visual responses
        responses = model.generate_responses(
            stimuli=stimuli,
            perspectives=perspectives,
            modulations=modulations,
        )
        responses = tqdm(responses, desc="Responses")
        responses = np.stack(list(responses), axis=0)

        # responses and trial_ids
        return responses, trial_ids
