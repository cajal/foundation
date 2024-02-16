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
        from foundation.recording.trial import (
            Trial,
            TrialSet,
            TrialVideo,
            TrialBounds,
            TrialFilterSet,
        )

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
                    logger.warning(
                        f"No trials found for video_id `{video['video_id']}`"
                    )
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
        np.array
            direction
        np.array
            mean response to direction
        np.array
            number of trials per direction
        """
        from foundation.recording.trace import Trace
        from foundation.recording.trial import Trial
        from foundation.stimulus.video import VideoSet, Video
        from foundation.utility.resample import Offset
        from foundation.utility.impulse import Impulse
        from foundation.utility.numeric import Precision
        from foundation.stimulus.compute.video import DirectionType

        # trial set
        trialset = (recording.TraceTrials & self.item).fetch1()

        # videos
        video_ids = (VideoSet & self.item).members
        video_ids = video_ids.fetch("KEY", order_by=video_ids.primary_key)

        # offset
        offset = (Offset & self.item).link.offset

        # get unit trace time and value
        trace = (Trace & self.item).link.compute
        trace_times, trace_values = trace.times, trace.values

        # initialize Impulse
        impulse = (Impulse & self.item).link.impulse(trace_times, trace_values, offset)

        # round for numeric precision
        round = (Precision & self.item).link.string

        with cache_rowproperty():
            subtrials = []

            for video_id in tqdm(video_ids, desc="Videos"):
                # check if video is a direction type
                video = (Video & video_id).link.compute
                assert isinstance(
                    video, DirectionType
                ), f"Video `{video_id['video_id']}` is not a direction type"

                # get direction onset and offset times within a single trial
                directions, dir_starts, dir_ends = list(zip(*video.directions()))
                directions = [round(direction) for direction in directions]
                dir_starts = np.array(dir_starts)
                dir_ends = np.array(dir_ends)

                # trial ids
                trial_ids = (VisualTrials & trialset & video_id & self.item).trial_ids

                # loop over trials
                for trial_id in trial_ids:
                    # get trial start and end
                    trial_start, trial_end = (
                        Trial & dict(trial_id=trial_id)
                    ).link.compute.bounds
                    # compute direction start and end times
                    subtrial_starts = dir_starts + trial_start
                    subtrial_ends = dir_ends + trial_start
                    subtrial_starts = np.minimum(subtrial_starts, trial_end)  # clip to trial end
                    subtrial_ends = np.minimum(subtrial_ends, trial_end)  # clip to trial end


                    for direction, start, end in zip(
                        directions, subtrial_starts, subtrial_ends
                    ):  
                        subtrials.append(
                            dict(
                                direction=direction,
                                response=impulse(start, end),  # returns nan if no response in time range
                            )
                        )

            # no trials at all
            if not subtrials:
                logger.warning("No trials found")

            subtrials = pd.DataFrame(subtrials)
            summary = subtrials.groupby("direction").agg(
                dict(
                    response=["mean", "count"],
                )
            )

            return (
                summary.index.to_list(),
                summary["response"]["mean"].to_numpy(),
                summary["response"]["count"].to_numpy(),
            )

if __name__ == '__main__':

    def test_visual_direction_tuning():
        from foundation.virtual import recording, stimulus, utility
        from foundation.recording.compute.visual import VisualDirectionTuning
        unit_key = recording.Trace.ScanUnit & 'animal_id=17797 and session=4 and scan_idx=7 and unit_id=1'
        trial_filter_set = recording.TrialFilterSet & 'trial_filterset_id="d00bbb175d63398818ca652391c18856"'
        video_set = stimulus.VideoSet & 'videoset_id="01eba4d8945512806337d42d015f2780"'
        offset = utility.Offset.MsOffset & 'ms_offset=150'
        test = VisualDirectionTuning & unit_key & trial_filter_set & video_set & offset
        return test.tuning

