import numpy as np
import pandas as pd
from itertools import repeat
from djutils import keys, rowproperty, cache_rowproperty
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
        videos = (VideoSet & self.item).members
        videos = videos.fetch("KEY", order_by=videos.primary_key)

        # trials, targets, predictions
        trials = []
        targs = []
        preds = []

        with cache_rowproperty():

            for video in tqdm(videos, desc="Videos"):

                # trials
                trial_ids = (VisualTrials & trialset & video & self.item).trial_ids

                # no trials for video
                if not trial_ids:
                    logger.warning(f"No trials found for video_id `{video['video_id']}`")
                    continue

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
class VisualDirectionTuning:
    """Visual Direction"""

    @property
    def keys(self):
        return [
            fnn.Model,
            stimulus.VideoSet,
            utility.Burnin,
            utility.Impulse,
            utility.Precision,
            utility.Offset,
        ]

    @rowproperty
    def tunings(self):
        """
        Returns
        -------
        numpy.ndarray
            directions presented (degrees), sorted
        numpy.ndarray
            list of unit-wise mean responses to directions, direction x units
        numpy.ndarray
            number of trials per direction
        """
        from foundation.utility.resample import Offset
        from foundation.utility.resize import Resize
        from foundation.utility.impulse import Impulse
        from foundation.stimulus.video import VideoSet, Video
        from foundation.stimulus.compute.video import DirectionType
        from foundation.fnn.model import Model
        from foundation.fnn.data import Data
        from foundation.utils import cuda_enabled
        from tqdm import tqdm

        # load model
        model = (Model & self.item).model(device="cuda" if cuda_enabled() else "cpu")

        # load data
        data = (Data & self.item).link.compute

        # model offset
        model_offset = data.unit_offset
        target_offset = (Offset & self.item).link.offset
        offset_correction = target_offset - model_offset

        # model period
        period = data.sampling_period

        # model response burnin
        burnin = self.item["burnin"]

        # videos
        video_ids = (VideoSet & self.item).members
        video_ids = video_ids.fetch("KEY", order_by=video_ids.primary_key)


        with cache_rowproperty():
            df = []
            stimuli = []
            current_time = 0

            # resize all stimuli and concatenate
            for video_id in tqdm(video_ids, desc="Videos"):
                
                # check if video is a direction type
                video = (Video & video_id).link.compute
                assert isinstance(video, DirectionType), "Video is not a direction type"

                # get direction onset and offset times within a single trial
                df.extend(
                    (round(_dir), current_time + _start, current_time + _end) for _dir, _start, _end in video.directions()
                )

                # get resized stimulus
                rvideo = (
                    Resize & {'resize_id': data.resize_id}
                ).link.resize(video.video, *data.resolution)
                stimuli.append(list(rvideo.generate(period=period, display_progress=False)))

                current_time += rvideo.period * len(rvideo)
                
            df = pd.DataFrame(df, columns=["direction", "start", "end"])
            stimuli = np.concatenate(stimuli, axis=0)

            # generate prediction
            r = model.generate_response(tqdm(stimuli))

            r = np.stack(list(r), axis=0)  # (frame, units)

            # compute time from period
            t = np.arange(len(r)) * period

            # impulse
            impulse = (Impulse & self.item).link.impulse(
                t[burnin:], r[burnin:], offset_correction
            )
            df["response"] = df.apply(
                lambda x: impulse(x["start"], x["end"]), axis=1
            )

        df = df.groupby("direction")['response'].agg(
            mean=lambda x: np.mean(x, axis=0),
            count=lambda x: len(x),
        )
        df.index = df.index.astype(float)
        df = df.reset_index()
        df = df.sort_values("direction")
        return (
            df['direction'].to_numpy(), 
            np.stack(df["mean"].to_numpy(), axis=0),  # direction x units
            df["count"].to_numpy()
        )
    

if __name__ == "__main__":

    def test_visual_direction_tuning():
        from foundation.virtual import recording, stimulus, utility
        from foundation.fnn.compute.visual import VisualDirectionTuning
        import os
        os.environ['FOUNDATION_CUDA'] = '1'

        model_key = {
            'data_id': '11e7be67a39d58be8a10202f654af2b3',
            'network_id': 'c17d459afa99a88b3e48a32fbabc21e4',
            'instance_id': '6600970e9cfe7860b80a70375cb6f20c'
        }
        video_set = stimulus.VideoSet & 'videoset_id="9df3d6b96304c6ed9807c27e0d46966c"'  # random 10 Monet2 video
        offset = utility.Offset.MsOffset & "ms_offset=150"
        burnin = utility.Burnin & "burnin=10"
        test = VisualDirectionTuning & model_key & video_set & offset & burnin
        return test.tunings
