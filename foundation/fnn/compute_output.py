import numpy as np
import pandas as pd
from djutils import keys, rowmethod, rowproperty
from foundation.virtual import utility, stimulus, recording, fnn


# ----------------------------- Network Output -----------------------------

# -- Network Output Types --


@keys
class NetworkOutput:
    """Model Output"""

    @property
    def key_list(self):
        return [
            fnn.ModelNetwork,
        ]

    @rowmethod
    def visual(self, video_id):
        """
        Parameters
        ----------
        video_id : str
            key (foundation.stimulus.video.Video)

        Returns
        -------
        2D array
            [samples, units] ; response traces
        """
        return (Visual & self.key & {"video_id": video_id}).output

    @rowmethod
    def visual_recording(self, video_id, trial_filterset_id, trial_perspectives=True, trial_modulations=True):
        """
        Parameters
        ----------
        video_id : str
            key (foundation.stimulus.video.Video)
        trial_filterset_id : str
            key (foundation.recording.trial.TrialFilterSet)
        trial_perspective : bool
            use recording trial perspective
        trial_modulation : bool
            use recording trial modulation

        Returns
        -------
        pandas.Series
            index -- str | None
                : trial_id -- key (foundation.recording.trial.Trial) | None
            data -- 2D array
                : [samples, units] ; response traces
        """
        key = VisualRecording & self.key
        key &= {
            "video_id": video_id,
            "trial_filterset_id": trial_filterset_id,
            "trial_perspectives": trial_perspectives,
            "trial_modulations": trial_modulations,
        }
        return key.output


# -- Network Output Intermediates --


@keys
class Visual:
    """Visual"""

    @property
    def key_list(self):
        return [
            fnn.ModelNetwork,
            stimulus.Video,
        ]

    @rowproperty
    def output(self):
        from foundation.fnn.network import Network
        from foundation.fnn.model import ModelNetwork

        # fetch attributes
        video_id = self.key.fetch1("video_id")

        # load model
        model = (ModelNetwork & self.key).model

        # load data
        data = (Network & self.key).link.data
        _, stimuli, _, _ = data.link.network_input.visual(video_id=video_id)

        # generate outputs
        outputs = model.generate_responses(stimuli=stimuli)
        return np.stack(list(outputs), 1).squeeze(0)


@keys
class VisualRecording:
    """Visual Recording Output"""

    @property
    def key_list(self):
        return [
            fnn.ModelNetwork,
            stimulus.Video,
            recording.TrialFilterSet,
            utility.Bool.proj(trial_perspectives="bool"),
            utility.Bool.proj(trial_modulations="bool"),
        ]

    @rowproperty
    def output(self):
        from foundation.fnn.network import Network
        from foundation.fnn.model import ModelNetwork

        # fetch attributes
        video_id, tset_id, persp, modul = self.key.fetch1(
            "video_id", "trial_filterset_id", "trial_perspectives", "trial_modulations"
        )

        # load model
        model = (ModelNetwork & self.key).model

        # load data
        data = (Network & self.key).link.data
        trial_ids, stimuli, perspectives, modulations = data.link.network_input.visual(
            video_id=video_id, trial_filterset_id=tset_id, trial_perspectives=persp, trial_modulations=modul
        )

        # generate outputs
        outputs = model.generate_responses(stimuli=stimuli, perspectives=perspectives, modulations=modulations)
        outputs = [*np.stack(list(outputs), 1)]

        # series index
        if not trial_ids:
            trial_ids.append(None)

        # output series
        return pd.Series(data=outputs, index=pd.Index(trial_ids, name="trial_id"))