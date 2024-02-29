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
        from foundation.stimulus.video import VideoSet, Video
        from foundation.stimulus.compute.video import DirectionType
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

        # videos
        videos = (VideoSet & self.item).members

        # trials
        trials = (VisualTrials & self.item & trialset & videos).trial_videos

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


@keys
class VisualSpatialGrid:
    """Visual Spatial Grid"""

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
            spatial_type -- spatial type
            spatial_grid -- spatial grid
        """
        from foundation.stimulus.video import VideoSet, Video
        from foundation.stimulus.compute.video import SpatialType
        from torch import tensor, nn

        # videos
        videos = (VideoSet & self.item).members

        # trials
        trials = (VisualTrials & self.item & videos).trial_videos

        # fetch trials
        trial_ids, video_ids, starts = trials.fetch("trial_id", "video_id", "start", order_by="start")

        # trial dataframe
        tdf = pd.DataFrame({"trial_id": trial_ids, "video_id": video_ids, "start": starts})

        # video dataframe
        rows = []
        for video_id in tqdm(np.unique(video_ids), desc="Videos"):

            # load video
            video = (Video & {"video_id": video_id}).link.compute
            assert isinstance(video, SpatialType)

            # spatial info
            stypes, sgrids, onsets, offsets = zip(*video.spatials())

            # resize spatial grids
            sgrids = tensor(np.stack(sgrids)[None])
            sgrids = nn.functional.interpolate(sgrids, [self.item["height"], self.item["width"]], mode="area")
            sgrids = sgrids[0].numpy()

            for stype, sgrid, onset, offset in zip(stypes, sgrids, onsets, offsets):

                row = {
                    "video_id": video_id,
                    "onset": onset,
                    "offset": offset,
                    "spatial_type": stype,
                    "spatial_grid": sgrid,
                }
                rows.append(row)

        vdf = pd.DataFrame(rows)

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
        df = (VisualSpatialGrid & trialset & self.item).df.copy()

        # spatial response
        df["response"] = df.apply(lambda x: impulse(x.start + x.onset, x.start + x.offset), axis=1)

        # drop NA response and sort by spatial type
        df = df[df.response.notna()].sort_values("spatial_type")

        # iterate spatial types
        for spatial_type, sdf in df.groupby("spatial_type"):

            # compute density and response
            grids = np.stack(sdf.spatial_grid, axis=-1)
            density = grids.sum(axis=-1)
            response = (grids * sdf.response.values).sum(axis=-1) / density

            yield spatial_type, response, density
