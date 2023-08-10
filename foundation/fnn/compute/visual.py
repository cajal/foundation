import numpy as np
from itertools import repeat
from djutils import keys, rowproperty, cache_rowproperty
from foundation.utils import tqdm
from foundation.virtual import utility, stimulus, recording, fnn


@keys
class VisualRecordingCorrelation:
    """Visual Recording Correlation"""

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
    def units(self):
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
        model = (Model & self.item).model(device="cuda" if cuda_enabled() else "cpu")

        # load data
        data = (Data & self.item).link.compute

        # trial set
        trialset = {"trialset_id": data.trialset_id}

        # videos
        videos = (VideoSet & self.item).members.fetch("video_id", order_by="video_id", as_dict=True)

        # trials, targets, predictions
        trials = []
        targs = []
        preds = []

        with cache_rowproperty():

            for video in tqdm(videos, desc="Videos"):

                # trials
                trial_ids = (VisualTrials & trialset & video & self.item).trial_ids
                trials.append(trial_ids)

                # stimuli
                stimuli = data.trial_stimuli(trial_ids)

                # units
                units = data.trial_units(trial_ids)

                # perspectives
                if self.item["perspective"]:
                    perspectives = data.trial_perspectives(trial_ids)
                else:
                    perspectives = repeat(None)

                # modulations
                if self.item["modulation"]:
                    modulations = data.trial_modulations(trial_ids)
                else:
                    modulations = repeat(None)

                # video targets and predictions
                _targs = []
                _preds = []

                for s, p, m, u in zip(stimuli, perspectives, modulations, units):

                    # generate prediction
                    r = model.generate_response(stimuli=s, perspectives=p, modulations=m)
                    r = np.stack(list(r), axis=0)

                    _targs.append(u)
                    _preds.append(r)

                assert len(_targs) == len(_preds) == len(trial_ids)

                targs.append(_targs)
                preds.append(_preds)

            assert len(targs) == len(preds) == len(videos)

        # correlations
        cc = (Correlation & self.item).link.correlation
        correlations = []

        for i in tqdm(range(data.units), desc="Units"):

            # unit targets and predictions
            unit_targ = []
            unit_pred = []

            for index, t, p in zip(trials, targs, preds):

                # target and prediction trials
                _unit_targ = Trials([_[:, i] for _ in t], index=index)
                _unit_pred = Trials([_[:, i] for _ in p], index=index)

                assert _unit_targ.matches(_unit_pred)

                unit_targ.append(_unit_targ)
                unit_pred.append(_unit_pred)

            # concatenated targets and predictions
            unit_targ = concatenate(*unit_targ, burnin=self.item["burnin"])
            unit_pred = concatenate(*unit_pred, burnin=self.item["burnin"])

            # unit correlation
            correlations.append(cc(unit_targ, unit_pred))

        return np.array(correlations)
