import numpy as np
import pandas as pd
from djutils import keys, rowproperty
from foundation.virtual import utility, stimulus, recording, fnn


# ----------------------------- Output -----------------------------

# -- Output Base --


class Output:
    """Model Output"""

    @rowproperty
    def outputs(self):
        """
        Returns
        -------
        pandas.Series
            index -- str | None
                : unique trial identifier | None
            data -- 2D array
                : [timepoints, units] ; response traces
        """
        raise NotImplementedError()


# -- Output Types --


@keys
class Visual(Output):
    """Visual Output"""

    @property
    def key_list(self):
        return [
            stimulus.Video,
            fnn.ModelNetwork,
        ]

    @rowproperty
    def outputs(self):
        from foundation.fnn.network import Network
        from foundation.fnn.model import ModelNetwork

        # fetch attributes
        video_id = self.key.fetch1("video_id")

        # load model
        model = (ModelNetwork & self.key).model

        # load data
        data = (Network & self.key).link.data
        data = data.link.visual_input
        _, stimuli, _, _ = data.inputs(video_id=video_id)

        # generate outputs
        outputs = model.generate_responses(stimuli=stimuli)
        outputs = np.stack(list(outputs), 1).squeeze(0)

        # output series
        return pd.Series(data=[outputs], index=[None])


@keys
class VisualRecording(Output):
    """Visual Recording Output"""

    @property
    def key_list(self):
        return [
            stimulus.Video,
            fnn.ModelNetwork,
            recording.TrialFilterSet,
            utility.Bool.proj(perspective="bool"),
            utility.Bool.proj(modulation="bool"),
        ]

    @rowproperty
    def outputs(self):
        from foundation.fnn.network import Network
        from foundation.fnn.model import ModelNetwork

        # fetch attributes
        video_id, tset_id, perspective, modulation = self.key.fetch1(
            "video_id", "trial_filterset_id", "perspective", "modulation"
        )

        # load model
        model = (ModelNetwork & self.key).model

        # load data
        data = (Network & self.key).link.data
        data = data.link.visual_input
        trial_ids, stimuli, perspectives, modulations = data.inputs(
            video_id=video_id, trial_filterset_id=tset_id, perspective=perspective, modulation=modulation
        )

        # generate outputs
        outputs = model.generate_responses(stimuli=stimuli, perspectives=perspectives, modulations=modulations)
        outputs = [*np.stack(list(outputs), 1)]

        # series index
        if not trial_ids:
            trial_ids.append(None)

        # output series
        return pd.Series(data=outputs, index=trial_ids)
