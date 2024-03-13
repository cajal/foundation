import numpy as np
import pandas as pd
from itertools import repeat
from djutils import keys, rowmethod, cache_rowproperty
from foundation.utils import tqdm, logger
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

    @rowmethod
    def correlation(self):
        """
        Returns
        -------
        1D array
            correlation values -- [units]
        """
        from foundation.fnn.model import Model
        from foundation.fnn.data import Data
        from foundation.utility.response import Correlation
        from foundation.recording.compute.visual import VisualTrialSet
        from foundation.utils.response import Trials, concatenate
        from foundation.utils import cuda_enabled

        # load model
        model = (Model & self.item).model(device="cuda" if cuda_enabled() else "cpu")

        # load data
        data = (Data & self.item).link.compute

        # trial set
        trialset = {"trialset_id": data.trialset_id}

        # trial df
        df = (VisualTrialSet & trialset & self.item).df

        # trials, targets, predictions
        trials = []
        targs = []
        preds = []

        with cache_rowproperty():

            for _, vdf in tqdm(df.groupby("video_id"), desc="Videos"):

                # trial ids
                trial_ids = list(vdf.trial_id)

                # trial stimuli
                stimuli = data.trial_stimuli(trial_ids)

                # trial units
                units = data.trial_units(trial_ids)

                # trial perspectives
                if self.item["perspective"]:
                    perspectives = data.trial_perspectives(trial_ids)
                else:
                    perspectives = repeat(None)

                # trial modulations
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

                assert len(trial_ids) == len(_targs) == len(_preds)

                trials.append(trial_ids)
                targs.append(_targs)
                preds.append(_preds)

        # no trials at all
        if not trials:
            logger.warning(f"No trials found")
            return

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


@keys
class VisualImpulse:
    """Visual Response"""

    @property
    def keys(self):
        return [
            fnn.Model,
            stimulus.VideoSet,
            utility.Offset,
            utility.Impulse,
            utility.Burnin,
        ]

    @rowmethod
    def impulse(self):
        """
        Yields
        ------
        str
            key (foundation.stimulus.video.Video)
        foundation.utils.impulse.Impulse
            impulse function
        """
        from foundation.fnn.model import Model
        from foundation.fnn.data import Data
        from foundation.stimulus.video import Video, VideoSet
        from foundation.utility.resample import Offset
        from foundation.utility.resize import Resize
        from foundation.utility.impulse import Impulse
        from foundation.utils import cuda_enabled

        # load model
        model = (Model & self.item).model(device="cuda" if cuda_enabled() else "cpu")

        # load data
        data = (Data & self.item).link.compute

        # stimulus resolution
        height, width = data.resolution

        # sampling period
        period = data.sampling_period

        # response offset
        offset = (Offset & self.item).link.offset - data.unit_offset

        # resize function
        resize = (Resize & {"resize_id": data.resize_id}).link.resize

        # impulse function
        impulse = (Impulse & self.item).link.impulse

        # videos
        video_ids = (VideoSet & self.item).members.fetch("video_id", order_by="video_id")

        for video_id in tqdm(video_ids, desc="Videos"):

            # load video
            video = resize(
                video=(Video & {"video_id": video_id}).link.compute.video,
                height=height,
                width=width,
            ).generate(period=period, display_progress=False)

            # response to video
            response = model.generate_response(video)
            response = np.stack(list(response), axis=0)

            # response impulse
            imp = impulse(
                times=np.arange(self.item["burnin"], len(response)) * period,
                values=response[self.item["burnin"] :],
                target_offset=offset,
            )

            yield video_id, imp


@keys
class VisualDirectionTuning:
    """Visual Direction"""

    @property
    def keys(self):
        return [
            fnn.Model,
            stimulus.VideoSet,
            utility.Offset,
            utility.Impulse,
            utility.Precision,
            utility.Burnin,
        ]

    @rowmethod
    def tuning(self):
        """
        Returns
        -------
        1D array
            directions (degrees) -- [directions]
        2D array
            response (STA) to directions -- [directions X units]
        2D array
            density of directions -- [directions X units]
        """
        from foundation.stimulus.video import VideoSet
        from foundation.stimulus.compute.video import DirectionSet
        from foundation.utility.numeric import Precision

        # precision function
        pstr = (Precision & self.item).link.string

        # videos
        videos = (VideoSet & self.item).members

        # directions
        directions = (DirectionSet & videos).df.groupby("video_id")

        # response dataframe
        dfs = []
        for video_id, impulse in (VisualImpulse & self.item).impulse():

            # direction dataframe
            df = directions.get_group(video_id)

            # direction response and discretization
            df.loc[:, ["response"]] = df.apply(lambda x: impulse(x.onset, x.offset), axis=1)
            df.loc[:, ["direction"]] = df.apply(lambda x: pstr(x.direction), axis=1)

            dfs.append(df)

        df = pd.concat(dfs)
        df = df.groupby("direction")["response"].agg(response=lambda x: np.stack(x, axis=1))

        # compute density and response STA
        df.loc[:, ["density"]] = df.apply(lambda x: np.isfinite(x.response).sum(axis=1), axis=1)
        df.loc[:, ["response"]] = df.apply(lambda x: np.nansum(x.response, axis=1) / x.density, axis=1)

        # sort by direction
        df.index = df.index.astype(float)
        df = df.reset_index()
        df = df.sort_values("direction")

        return (
            df["direction"].values,
            np.stack(df["response"], axis=1).astype(np.float32),
            np.stack(df["density"], axis=1).astype(int),
        )
