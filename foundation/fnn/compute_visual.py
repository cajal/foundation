import numpy as np
from djutils import keys, rowproperty
from foundation.utils import tqdm
from foundation.virtual import utility, stimulus, fnn


@keys
class NetworkModel:
    """Visual Network Model"""

    @property
    def key_list(self):
        return [
            stimulus.Video,  # visual stimulus
            fnn.NetworkModel,  # network model
            utility.Bool.proj(perspectives="bool"),  # True (trial perspectives)| False (default perspective)
            utility.Bool.proj(modulations="bool"),  # True (trial modulations) | False (default modulation)
            utility.Bool.proj(trainset="bool"),  # True (trainset trials) | False (testset trials)
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
        video_id, perspectives, modulations, trainset = self.key.fetch1(
            "video_id", "perspectives", "modulations", "trainset"
        )
        stimuli, perspectives, modulations, trial_ids = (Network & self.key).link.data.visual_inputs(
            video_id=video_id,
            perspectives=perspectives,
            modulations=modulations,
            trainset=trainset,
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
