import numpy as np
import datajoint as dj
from djutils import merge, skip_missing
from tqdm import tqdm
from foundation.recording import resample
from foundation.schemas.pipeline import (
    pipe_exp,
    pipe_shared,
    pipe_stim,
    pipe_fuse,
    pipe_meso,
    pipe_reso,
    pipe_eye,
    pipe_tread,
)
from foundation.schemas import scan as schema


@schema
class Times(dj.Computed):
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
        pipe = dj.U("pipe") & (pipe_fuse.ScanDone & key)
        pipe = pipe.fetch1("pipe")

        if pipe == "meso":
            pipe = pipe_meso
        elif pipe == "reso":
            pipe = pipe_reso
        else:
            raise ValueError(f"{pipe} not recognized")

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


@schema
class Pupil(dj.Computed):
    definition = """
    -> pipe_eye.FittedPupil
    pupil_type      : enum("radius", "center_x", "center_y")    # pupil data type
    ---
    pupil_trace     : longblob  # pupil trace
    """

    def make(self, key):
        fits = pipe_eye.FittedPupil.Circle & key

        radius, center = fits.fetch("radius", "center", order_by="frame_id")
        center = [np.array([np.nan, np.nan]) if c is None else c for c in center]
        x, y = np.array(center).T

        keys = [
            dict(key, pupil_type="radius", pupil_trace=radius),
            dict(key, pupil_type="center_x", pupil_trace=x),
            dict(key, pupil_type="center_y", pupil_trace=y),
        ]
        self.insert(keys)


# ---------- Populate Functions ----------


def populate_scan(
    animal_id,
    session,
    scan_idx,
    pipe_version=1,
    segmentation_method=6,
    classification_method=2,
    unit_type="soma",
    spike_method=6,
    tracking_method=2,
    reserve_jobs=True,
    display_progress=True,
):
    """
    Parameters
    ----------
    animal_id : int
        animal id
    session : int
        scan session
    scan_idx : int
        scan index
    ...
    """
    from foundation.stimulus import video
    from foundation.recording import trial, trace

    # scan
    key = dict(
        animal_id=animal_id,
        session=session,
        scan_idx=scan_idx,
    )
    pipe = pipeline(**key)

    # populate bounds
    ScanTimes.populate(key, reserve_jobs=reserve_jobs, display_progress=display_progress)
    EyeTimes.populate(key, reserve_jobs=reserve_jobs, display_progress=display_progress)
    TreadmillTimes.populate(key, reserve_jobs=reserve_jobs, display_progress=display_progress)

    # scan trials
    scan_trials = pipe_stim.Trial & key

    # stimulus types
    conditions = pipe_stim.Condition & scan_trials
    stim_types = dj.U("stimulus_type") & conditions
    stim_types = stim_types.fetch("stimulus_type")

    # fill stimulus types
    for stim_type in stim_types:

        table = stim_type.split(".")[1]
        table = getattr(video, table, None)

        if table is None:
            raise NotImplementedError(f"Condition {stim_type} is not yet implemented")
        else:
            conds = conditions & dict(stimulus_type=stim_type)
            table.insert(conds.proj(), skip_duplicates=True)

    # populate video
    video.VideoLink.fill()
    video.VideoInfo.populate(reserve_jobs=reserve_jobs, display_progress=display_progress)

    # populate trial
    trial.ScanTrial.insert(scan_trials.proj(), skip_duplicates=True)
    trial.TrialLink.fill()
    trial.TrialBounds.populate(reserve_jobs=reserve_jobs, display_progress=display_progress)
    trial.TrialSamples.populate(reserve_jobs=reserve_jobs, display_progress=display_progress)
    trial.TrialVideo.populate(reserve_jobs=reserve_jobs, display_progress=display_progress)
    trial.VideoSamples.populate(reserve_jobs=reserve_jobs, display_progress=display_progress)

    # fill unit traces
    _key = dict(
        key,
        pipe_version=pipe_version,
        segmentation_method=segmentation_method,
        classification_method=classification_method,
        type=unit_type,
    )
    units = pipe.ScanSet.Unit & (pipe.MaskClassification.Type & _key)
    traces = pipe_meso.Activity.Trace & units.proj() & dict(key, spike_method=spike_method)
    trace.MesoActivity.insert(traces.proj(), skip_duplicates=True)

    # fill pupil traces
    pupil_types = ["radius", "center_x", "center_y"]
    _key = [dict(key, tracking_method=tracking_method, pupil_type=p) for p in pupil_types]
    trace.ScanPupil.insert(_key, skip_duplicates=True)

    # fill treadmill trace
    trace.ScanTreadmill.insert1(key, skip_duplicates=True)

    # # populate traces
    trace.TraceLink.fill()
    trace.TraceTrials.populate(reserve_jobs=reserve_jobs, display_progress=display_progress)
    trace.TraceBounds.populate(reserve_jobs=reserve_jobs, display_progress=display_progress)
