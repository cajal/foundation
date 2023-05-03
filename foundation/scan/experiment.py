import numpy as np
import datajoint as dj
from foundation.utils.logging import logger
from foundation.schemas.pipeline import (
    pipe_exp,
    pipe_stim,
    pipe_eye,
    pipe_tread,
    resolve_pipe,
)
from foundation.schemas import scan as schema


@schema.computed
class Scan:
    definition = """
    -> pipe_exp.Scan
    ---
    scan_times          : longblob      # scan trace times on the stimulus clock
    eye_times           : longblob      # eye trace times on the stimulus clock
    treadmill_times     : longblob      # treadmill trace times on the stimulus clock
    """

    def make(self, key):
        from scipy.interpolate import interp1d

        # resolve pipeline
        pipe = resolve_pipe(key)

        # number of planes
        n = (pipe.ScanInfo & key).proj(n="nfields div nrois").fetch1("n")
        if n != len(dj.U("z") & (pipe.ScanInfo.Field & key)):
            raise ValueError("unexpected number of depths")

        # fetch and slice times
        stim_times = (pipe_stim.Sync & key).fetch1("frame_times", squeeze=True)[::n]
        beh_times = (pipe_stim.BehaviorSync & key).fetch1("frame_times", squeeze=True)[::n]

        assert len(stim_times) == len(beh_times)
        assert np.isfinite(stim_times).all()
        assert np.isfinite(beh_times).all()

        # truncate times
        nframes = (pipe.ScanInfo & key).fetch1("nframes")
        assert 0 <= len(stim_times) - nframes <= 1

        stim_times = stim_times[:nframes]
        beh_times = beh_times[:nframes]

        # median times
        stim_median = np.median(stim_times)
        beh_median = np.median(beh_times)

        # behavior -> stimulus clock
        beh_to_stim = interp1d(
            x=beh_times - beh_median,
            y=stim_times - stim_median,
            kind="linear",
            fill_value=np.nan,
            bounds_error=False,
            copy=False,
        )

        # eye times
        raw = (pipe_eye.Eye & key).fetch1("eye_time")
        nans = np.isnan(raw)
        eye_times = np.full_like(raw, np.nan)
        eye_times[~nans] = beh_to_stim(raw[~nans] - beh_median) + stim_median

        # treadmill times
        raw = (pipe_tread.Treadmill & key).fetch1("treadmill_time")
        nans = np.isnan(raw)
        tread_times = np.full_like(raw, np.nan)
        tread_times[~nans] = beh_to_stim(raw[~nans] - beh_median) + stim_median

        # insert key
        key["scan_times"] = stim_times
        key["eye_times"] = eye_times
        key["treadmill_times"] = tread_times
        self.insert1(key)

    def fill_videos(self):
        from foundation.stimulus.video import VideoLink, VideoInfo

        # scan trials
        trials = pipe_stim.Trial * pipe_stim.Condition & self

        # stimulus types
        stim_types = dj.U("stimulus_type") & trials
        for stim_type in stim_types.fetch("stimulus_type"):

            keys = trials & dict(stimulus_type=stim_type)
            table = VideoLink.get(stim_type.split(".")[1]).link
            table.insert(keys.proj(), skip_duplicates=True, ignore_extra_fields=True)

        # video link
        VideoLink.fill()

        # compute video
        VideoInfo.populate(display_progress=True, reserve_jobs=True)

    def fill_trials(self):
        from foundation.recording.trial import ScanTrial, TrialLink, TrialBounds, TrialVideo

        # scan trials
        trials = pipe_stim.Trial & self
        ScanTrial.insert(trials.proj(), skip_duplicates=True)

        # trial link
        TrialLink.fill()

        # computed trials
        key = TrialLink.ScanTrial & trials
        TrialBounds.populate(key, display_progress=True, reserve_jobs=True)
        TrialVideo.populate(key, display_progress=True, reserve_jobs=True)

    def fill_treadmill(self):
        from foundation.recording.trace import ScanTreadmill, TraceLink, TraceHomogeneous, TraceTrials

        # scan treadmill trace
        tread = pipe_tread.Treadmill & self
        ScanTreadmill.insert(tread.proj(), skip_duplicates=True)

        # trace link
        TraceLink.fill()

        # compute trace
        key = TraceLink.ScanTreadmill & tread
        TraceHomogeneous.populate(key, display_progress=True, reserve_jobs=True)
        TraceTrials.populate(key, display_progress=True, reserve_jobs=True)
