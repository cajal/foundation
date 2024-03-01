import numpy as np
import pandas as pd
from tqdm import tqdm
from djutils import keys, merge, rowproperty, cache_rowproperty, keyproperty, U
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

    @keyproperty(recording.TrialSet, recording.TrialFilterSet)
    def trial_videos(self):
        """
        Returns
        -------
        datajoint.Table
            trial_id | start, end, video_id
        """
        from foundation.recording.trial import Trial, TrialSet, TrialVideo, TrialBounds, TrialFilterSet

        # trials key
        key = (U("trialset_id", "trial_filterset_id") & self.key).fetch1("KEY")

        # all trials
        trials = Trial & (TrialSet & key).members

        # filtered trials
        trials = (TrialFilterSet & key).filter(trials)

        # video trials
        trials = merge(trials.proj(), TrialBounds, TrialVideo) & self.key

        return trials


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
class VisualDirectionSet:
    """Visual Direction Set"""

    @property
    def keys(self):
        return [
            recording.TrialSet,
            recording.TrialFilterSet,
            stimulus.VideoSet,
        ]

    @rowproperty
    def df(self):
        """
        Returns
        -------
        pd.DataFrame
            trial_id -- foundation.recording.trial.Trial
            video_id -- foundation.stimulus.video.Video
            start -- time of trial start (seconds)
            onset -- time of spatial onset (seconds relative to start)
            offset -- time of spatial offset (seconds relative to start)
            direction -- direction (degrees, 0 to 360)
        """
        from foundation.stimulus.video import VideoSet
        from foundation.stimulus.compute.video import DirectionSet

        # videos
        videos = (VideoSet & self.item).members

        # trials
        trials = (VisualTrials & self.item & videos).trial_videos

        # fetch trials
        trial_ids, video_ids, starts = trials.fetch("trial_id", "video_id", "start", order_by="start")

        # trial dataframe
        tdf = pd.DataFrame({"trial_id": trial_ids, "video_id": video_ids, "start": starts})

        # video dataframe
        vdf = (DirectionSet & videos).df()

        return tdf.merge(vdf)


@keys
class VisualDirectionTuning:
    """Visual Direction Tuning"""

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
        from foundation.recording.trace import Trace
        from foundation.utility.resample import Offset
        from foundation.utility.impulse import Impulse
        from foundation.utility.numeric import Precision

        # trace times and values
        trace = (Trace & self.item).link.compute
        times, values = trace.times, trace.values

        # offset
        offset = (Offset & self.item).link.offset

        # impulse
        impulse = (Impulse & self.item).link.impulse(times, values, offset)

        # precision
        pstr = (Precision & self.item).link.string

        # trialset
        trialset = (recording.TraceTrials & self.item).fetch1()

        # trial and video dataframe
        df = (VisualDirectionSet & trialset & self.item).df.copy()

        # direction response
        df["response"] = df.apply(lambda x: impulse(x.start + x.onset, x.start + x.offset), axis=1)

        # direction discretization
        df["direction"] = df.apply(lambda x: pstr(x.direction), axis=1)

        # drop NA response
        df = df[df.response.notna()]

        # compute response and density
        df = df.groupby("direction")
        df = df.agg(dict(response=["mean", "count"]))
        df.index = df.index.astype(float)
        df = df.reset_index()
        df = df.sort_values("direction")

        return df["direction"].values, df["response"]["mean"].values, df["response"]["count"].values


@keys
class VisualSpatialSet:
    """Visual Spatial Set"""

    @property
    def keys(self):
        return [
            recording.TrialSet,
            recording.TrialFilterSet,
            stimulus.VideoSet,
            utility.Resolution,
        ]

    @rowproperty
    def df(self):
        """
        Returns
        -------
        pd.DataFrame
            trial_id -- foundation.recording.trial.Trial
            video_id -- foundation.stimulus.video.Video
            start -- time of trial start (seconds)
            onset -- time of spatial onset (seconds relative to start)
            offset -- time of spatial offset (seconds relative to start)
            spatial_type -- spatial type (str)
            spatial_grid -- spatial grid (2D array)
        """
        from foundation.stimulus.video import VideoSet
        from foundation.stimulus.compute.video import SpatialSet

        # videos
        videos = (VideoSet & self.item).members

        # trials
        trials = (VisualTrials & self.item & videos).trial_videos

        # fetch trials
        trial_ids, video_ids, starts = trials.fetch("trial_id", "video_id", "start", order_by="start")

        # trial dataframe
        tdf = pd.DataFrame({"trial_id": trial_ids, "video_id": video_ids, "start": starts})

        # video dataframe
        vdf = (SpatialSet & videos).df(height=self.item["height"], width=self.item["width"])

        return tdf.merge(vdf)


@keys
class VisualSpatialTuning:
    """Visual Spatial Tuning"""

    @property
    def keys(self):
        return [
            recording.Trace,
            recording.TrialFilterSet,
            stimulus.VideoSet,
            utility.Offset,
            utility.Impulse,
            utility.Resolution,
        ]

    @rowproperty
    def tuning(self):
        """
        Yields
        ------
        str
            spatial type
        2D array
            response (STA) to spatial locations
        2D array
            density of spatial locations
        """
        from foundation.recording.trace import Trace
        from foundation.utility.resample import Offset
        from foundation.utility.impulse import Impulse

        # trace times and values
        trace = (Trace & self.item).link.compute
        times, values = trace.times, trace.values

        # offset
        offset = (Offset & self.item).link.offset

        # impulse
        impulse = (Impulse & self.item).link.impulse(times, values, offset)

        # trialset
        trialset = (recording.TraceTrials & self.item).fetch1()

        # trial and video dataframe
        df = (VisualSpatialSet & trialset & self.item).df.copy()

        # spatial response
        df["response"] = df.apply(lambda x: impulse(x.start + x.onset, x.start + x.offset), axis=1)

        # drop NA responses
        df = df[df.response.notna()]

        # iterate spatial types
        for spatial_type, sdf in df.groupby("spatial_type"):

            # compute density and response
            grids = np.stack(sdf.spatial_grid, axis=-1)
            density = grids.sum(axis=-1)
            response = (grids * sdf.response.values).sum(axis=-1) / density

            yield spatial_type, response, density
