import numpy as np
from scipy.interpolate import interp1d
from djutils import keys, merge, row_property
from foundation.utils.resample import frame_index
from foundation.stimulus.video import VideoInfo
from foundation.recording.trial import TrialLink, TrialBounds, TrialVideo
from foundation.utility.resample import RateLink, OffsetLink, ResampleLink


@keys
class ResampledTrialVideo:
    keys = [TrialLink, RateLink]

    @row_property
    def index(self):
        # flip times and resampling period
        flips = (TrialLink & self.key).link.flips
        period = (RateLink & self.key).link.period

        # trial and video info
        info = merge(self.key, TrialBounds, TrialVideo, VideoInfo)
        start, frames = info.fetch1("start", "frames")

        if len(flips) != frames:
            raise ValueError("Flips do not match video frames.")

        # sample index for each flip
        index = frame_index(flips - start, period)
        samples = np.arange(index[-1] + 1)

        # first flip of each sampling index
        first = np.diff(index, prepend=-1) > 0

        # for each of the samples, get the previous flip/video index
        previous = interp1d(
            x=index[first],
            y=np.where(first)[0],
            kind="previous",
        )
        return previous(samples).astype(int)
