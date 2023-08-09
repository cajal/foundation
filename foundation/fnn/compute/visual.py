import numpy as np
from itertools import repeat
from djutils import keys, rowproperty, cache_rowproperty
from foundation.utils import tqdm
from foundation.virtual import utility, stimulus, recording, fnn


@keys
class ModelRecordingCorrelations:
    """Model Recording"""

    @property
    def keys(self):
        return [
            fnn.Model,
            recording.TrialFilterSet,
            stimulus.VideoSet,
            utility.Correlation,
            utility.Burnin,
            utility.Bool.proj(perspective="bool"),
            utility.Bool.proj(modulation="bool"),
        ]

    @rowproperty
    def correlations(self):
        """
        Returns
        -------
        1D array
            [units] -- unitwise correlations
        """
        from foundation.recording.compute.visual import VisualTrials
        from foundation.utility.response import Correlation
        from foundation.stimulus.video import VideoSet
        from foundation.fnn.model import Model
        from foundation.fnn.data import Data
        from foundation.utils.response import Trials, concatenate
        from foundation.utils import cuda_enabled

        # load model
        model = (Model & key).model(device="cuda" if cuda_enabled() else "cpu")

        # load data
        data = (Data & key).link.compute

        # trial set
        trialset = {"trialset_id": data.trialset_id}

        # videos
        videos = (VideoSet & key).members.fetch("video_id", order_by="video_id", as_dict=True)

        # trials, targets, predictions
        trials = []
        targs = []
        preds = []

        with cache_rowproperty():

            for video in tqdm(videos, desc="Videos"):

                # trials
                trial_ids = (VisualTrials & trialset & video & key).trial_ids
                trials.append(trial_ids)

                # stimuli
                stimuli = data.trial_stimuli(trial_ids)

                # units
                units = data.trial_units(trial_ids)

                # perspectives
                if key["perspective"]:
                    perspectives = data.trial_perspectives(trial_ids)
                else:
                    perspectives = repeat([None])

                # modulations
                if key["modulation"]:
                    modulations = data.trial_modulations(trial_ids)
                else:
                    modulations = repeat([None])

                # video targets and predictions
                _targs = []
                _preds = []

                for s, p, m, u in zip(stimuli, perspectives, modulations, units):

                    # generate prediction
                    r = model.generate_response(stimuli=s, perspectives=p, modulations=m)
                    r = np.stack(list(r), axis=0)

                    # append target and prediction
                    _targs.append(u)
                    _preds.append(r)

                assert len(_targs) == len(_preds) == len(trial_ids)

                # append video targets and predictions
                targs.append(_targs)
                preds.append(_preds)

            assert len(targs) == len(preds) == len(videos)

        # correlations
        cc = (Correlation & key).link.correlation
        correlations = []

        for i in tqdm(range(data.units), desc="Units"):

            # unit targets and predictions
            unit_targ = []
            unit_pred = []

            for index, t, p in zip(trials, targs, preds):

                # target and prediction trials
                unit_targ.append(Trials([_[:, i] for _ in t], index=index))
                unit_pred.append(Trials([_[:, i] for _ in p], index=index))

                assert unit_targ.matches(unit_pred)

            # concatenated targets and predictions
            unit_targ = concatenate(*unit_targ, burnin=key["burnin"])
            unit_pred = concatenate(*unit_pred, burnin=key["burnin"])

            # unit correlation
            correlations.append(cc(unit_targ, unit_pred))

        return np.array(correlations)
