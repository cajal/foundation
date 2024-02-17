import numpy as np
import pandas as pd
from djutils import keys, merge, rowproperty, cache_rowproperty, MissingError
from foundation.utils import tqdm, logger
from foundation.virtual import utility, stimulus, recording


@keys
class VisualTrials:
    """Visual Trials"""

    @property
    def keys(self):
        return [
            recording.TrialSet,
            recording.TrialFilterSet,
            stimulus.Video,
        ]

    @rowproperty
    def trial_ids(self):
        """
        Returns
        -------
        Tuple[str]
            tuple of keys (foundation.recording.trial.Trial) -- ordered by trial start time
        """
        from foundation.recording.trial import Trial, TrialSet, TrialVideo, TrialBounds, TrialFilterSet

        # all trials
        trials = Trial & (TrialSet & self.item).members

        # filtered trials
        trials = (TrialFilterSet & self.item).filter(trials)

        # video trials
        trials = merge(trials, TrialBounds, TrialVideo) & self.item

        # trial ids, ordered by trial start
        return tuple(trials.fetch("trial_id", order_by="start"))


@keys
class VisualMeasure:
    """Visual Measure"""

    @property
    def keys(self):
        return [
            recording.Trace,
            recording.TrialFilterSet,
            stimulus.VideoSet,
            utility.Resample,
            utility.Offset,
            utility.Rate,
            utility.Measure,
            utility.Burnin,
        ]

    @rowproperty
    def measure(self):
        """
        Returns
        -------
        float
            visual response measure
        """
        from foundation.recording.compute.resample import ResampledTrace
        from foundation.stimulus.video import VideoSet
        from foundation.utility.response import Measure
        from foundation.utils.response import Trials, concatenate

        # trial set
        trialset = (recording.TraceTrials & self.item).fetch1()

        # videos
        videos = (VideoSet & self.item).members
        videos = videos.fetch("KEY", order_by=videos.primary_key)

        with cache_rowproperty():
            # visual responses
            responses = []

            for video in tqdm(videos, desc="Videos"):
                # trial ids
                trial_ids = (VisualTrials & trialset & video & self.item).trial_ids

                # no trials for video
                if not trial_ids:
                    logger.warning(f"No trials found for video_id `{video['video_id']}`")
                    continue

                # trial responses
                trials = (ResampledTrace & self.item).trials(trial_ids=trial_ids)
                trials = Trials(trials, index=trial_ids, tolerance=1)

                # append
                responses.append(trials)

        # no trials at all
        if not responses:
            logger.warning(f"No trials found")
            return

        # concatenated responses
        responses = concatenate(*responses, burnin=self.item["burnin"])

        # response measure
        return (Measure & self.item).link.measure(responses)


@keys
class VisualDirectionTuning:
    """Visual Direction"""

    @property
    def keys(self):
        return [
            recording.Trace,
            recording.TrialFilterSet,
            stimulus.VideoSet,
            utility.Offset,
            utility.Impulse,
            utility.Precision,
        ]

    @rowproperty
    def tuning(self):
        """
        Returns
        -------
        list
            direction
        list
            mean response to direction
        list
            number of trials per direction
        """
        from foundation.recording.trial import Trial, TrialSet, TrialVideo, TrialBounds, TrialFilterSet
        from foundation.recording.trace import Trace
        from foundation.stimulus.video import VideoSet, Video
        from foundation.utility.resample import Offset
        from foundation.utility.impulse import Impulse
        from foundation.utility.numeric import Precision
        from foundation.stimulus.compute.video import DirectionType

        # trace times and values
        trace = (Trace & self.item).link.compute
        times, values = trace.times, trace.values

        # offset
        offset = (Offset & self.item).link.offset

        # impulse
        impulse = (Impulse & self.item).link.impulse(times, values, offset)

        # precision
        pstr = (Precision & self.item).link.string

        # videos
        videos = (VideoSet & self.item).members

        # trial set
        trialset = (recording.TraceTrials & self.item).fetch1()

        # all trials
        trials = Trial & (TrialSet & trialset).members

        # filtered trials
        trials = (TrialFilterSet & self.item).filter(trials)

        # trial info
        trials = merge(trials, TrialBounds, TrialVideo)

        # trials restricted by videos
        trials = trials & videos

        # fetch trials
        video_ids, starts = trials.fetch("video_id", "start", order_by="start")

        # video dataframe
        rows = []
        for video_id in tqdm(np.unique(video_ids), desc="Videos"):

            video = (Video & {"video_id": video_id}).link.compute
            assert isinstance(video, DirectionType)

            dirs, ons, offs = zip(*video.directions())

            row = {"video_id": video_id, "dirs": list(map(pstr, dirs)), "ons": ons, "offs": offs}
            rows.append(row)

        vdf = pd.DataFrame(rows).set_index("video_id")

        # response dataframe
        rows = []
        for video_id, start in zip(tqdm(video_ids, desc="Trials"), starts):

            vid = vdf.loc[video_id]

            for direction, on, off in zip(vid.dirs, vid.ons, vid.offs):

                row = {"direction": direction, "response": impulse(start + on, start + off)}
                rows.append(row)

        rdf = pd.DataFrame(rows)
        rdf = rdf.groupby("direction")
        rdf = rdf.agg(dict(response=["mean", "count"]))
        rdf.index = rdf.index.astype(float)
        rdf = rdf.reset_index()
        rdf = rdf.sort_values("direction")

        return rdf["direction"].values, rdf["response"]["mean"].values, rdf["response"]["count"].values


if __name__ == "__main__":

    def test_visual_direction_tuning():
        from foundation.virtual import recording, stimulus, utility
        from foundation.recording.compute.visual import VisualDirectionTuning

        unit_key = recording.Trace.ScanUnit & "animal_id=17797 and session=4 and scan_idx=7 and unit_id=1"
        trial_filter_set = recording.TrialFilterSet & 'trial_filterset_id="d00bbb175d63398818ca652391c18856"'
        video_set = stimulus.VideoSet & 'videoset_id="01eba4d8945512806337d42d015f2780"'
        offset = utility.Offset.MsOffset & "ms_offset=150"
        test = VisualDirectionTuning & unit_key & trial_filter_set & video_set & offset
        return test.tuning
