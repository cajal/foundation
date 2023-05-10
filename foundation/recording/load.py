import numpy as np
import pandas as pd
from datajoint import U
from djutils import keys, merge, keyproperty
from foundation.utils.data import NpyFile
from foundation.utility.resize import Resize, Resolution
from foundation.utility.resample import Rate, Offset, Resample
from foundation.utility.standardize import Standardize
from foundation.stimulus.video import VideoInfo
from foundation.stimulus.cache import ResizedVideo
from foundation.recording.trial import Trial, TrialSet, TrialVideo
from foundation.recording.trace import TraceSet, TraceFilterSet, TraceSummary
from foundation.recording.cache import ResampledVideo, ResampledTraces
from foundation.recording.compute import StandardizeTraces


class Data:
    @property
    def load(self):
        raise NotImplementedError

    @property
    def shape(self):
        raise NotImplementedError


@keys
class Video(Data):
    """Load video stimulus"""

    @property
    def key_list(self):
        return [
            Trial,
            Resize,
            Resolution,
            Rate,
        ]

    @keyproperty(Resize, Resolution)
    def load(self):
        """
        Returns
        ----------
        pandas.Series
            index -- trial_id (foundation.recording.trial.Trial)
            data -- NpyFile (4D array [samples, height, width, channels])
        """
        # data tuples
        data = merge(self.key, TrialVideo, ResizedVideo, ResampledVideo)

        # fetch data
        trial_ids, filepaths, indexmaps = data.fetch("trial_id", "video", "index", order_by="trial_id")
        indexmaps = map(np.load, indexmaps)

        # data series
        return pd.Series(
            data=[NpyFile(filepath=f, indexmap=i) for f, i in zip(filepaths, indexmaps)],
            index=pd.Index(trial_ids, name="trial_id"),
        )

    @keyproperty(Resolution)
    def shape(self):
        """
        Returns
        ----------
        list[int]
            [height, width, channels]
        """
        key = merge(self.key, TrialVideo * VideoInfo)
        shape = U("height", "width", "channels") & key
        return list(shape.fetch1("height", "width", "channels"))


@keys
class Traces(Data):
    """Load recording traces"""

    @property
    def key_list(self):
        return [
            Trial,
            TraceSet,
            TrialSet,
            Standardize,
            Resample,
            Offset,
            Rate,
        ]

    @keyproperty(TraceSet, TrialSet, Standardize, Resample, Offset, Rate)
    def load(self):
        """
        Returns
        ----------
        pandas.Series
            index -- trial_id (foundation.recording.trial.Trial)
            data -- NpyFile (2D array [samples, traces])
        """
        # data standardization
        transform = (StandardizeTraces & self.key).transform

        # data tuples
        data = merge(self.key, ResampledTraces & "finite")

        # fetch data
        trial_ids, filepaths = data.fetch("trial_id", "traces")

        # data series
        return pd.Series(
            data=[NpyFile(filepath=f, transform=transform) for f in filepaths],
            index=pd.Index(trial_ids, name="trial_id"),
        )

    @keyproperty(TraceSet)
    def shape(self):
        """
        Returns
        ----------
        list[int]
            [traces]
        """
        return [(TraceSet & self.key).fetch1("members")]
