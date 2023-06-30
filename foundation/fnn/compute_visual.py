import numpy as np
from djutils import keys, rowproperty, cache_rowproperty, MissingError
from foundation.utils import tqdm
from foundation.virtual import utility, stimulus, recording, fnn


@keys
class VisualNetworkRecording:
    """Visual Network Model using data from Recording"""

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
    def trial_responses(self):
        """
        Returns
        -------
        3D array
            [trials, samples, units] -- visual response
        Tuple[str] | None
            tuple of trial_ids -- key (foundation.recording.trial.Trial)
        """
        from foundation.fnn.data import Data
        from foundation.fnn.network import Network
        from foundation.fnn.model import NetworkModel
        from foundation.utils.resample import truncate

        # load data
        data_id = (Network & self.item).link.data_id
        data = (Data & {"data_id": data_id}).link.compute

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
        responses = np.stack(list(responses), axis=1)

        if perspectives is None and modulations is None:
            # unsqueeze and repeat trials
            responses = np.expand_dims(responses, axis=0)
            responses = np.repeat(responses, repeats=len(trial_ids), axis=0)

        return responses, trial_ids


@keys
class VisualUnitCorrelation:
    """Correlation between Modeled and Recorded Unit"""

    @property
    def keys(self):
        return [
            fnn.NetworkModel,  # network model
            fnn.NetworkUnit,  # network unit
            utility.Bool.proj(trial_perspective="bool"),  # trial | default perspective
            utility.Bool.proj(trial_modulation="bool"),  # trial | default modulation
            recording.TrialFilterSet,  # trial filter
            stimulus.VideoSet,  # visual stimulus set
            utility.Correlation,  # correlation between model and recording
            utility.Burnin,  # response burnin frames
        ]

    @rowproperty
    def correlation(self):
        """
        Returns
        -------
        float
            correlation between model and target
        """
        from foundation.fnn.data import Data
        from foundation.fnn.network import Network
        from foundation.stimulus.video import VideoSet
        from foundation.utility.response import Correlation
        from foundation.utils.response import Trials, concatenate

        # unit index and burnin
        unit_index, burnin = self.key.fetch1("unit_index", "burnin")

        # videos
        videos = (VideoSet & self.item).members.fetch("video_id", as_dict=True)

        # data
        data_id = (Network & self.item).link.data_id
        data = (Data & {"data_id": data_id}).link.compute

        # model responses and targets
        responses = []
        targets = []

        with cache_rowproperty():
            for video in tqdm(videos, desc="Videos"):

                # model respose
                response, trial_ids = (VisualNetworkRecording & video & self.item).trial_responses
                response = Trials(response[:, unit_index], index=trial_ids, tolerance=0)

                # target
                target = data.trial_units(trial_ids, unit_index=unit_index)
                target = Trials(target, index=trial_ids, tolerance=1)

                # verify match
                assert response.matches(target)

                # append
                responses.append(response)
                targets.append(target)

        # concatenate model responses and targets
        responses = concatenate(*responses, burnin=self.item["burnin"])
        targets = concatenate(*targets, burnin=self.item["burnin"])

        # compute correlation
        return (Correlation & self.item).link.correlation(responses, targets)
