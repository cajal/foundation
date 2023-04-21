import numpy as np
import datajoint as dj
from scipy.interpolate import interp1d


# ---------- Populate Functions ----------


def populate_scan(animal_id, session, scan_idx, reserve_jobs=True, display_progress=True):
    """
    Parameters
    ----------
    animal_id : int
        animal id
    session : int
        scan session
    scan_idx : int
        scan index
    reserve_jobs : bool
        job reservation for AutoPopulate
    display_progress : bool
        display progress
    """
    from foundation.bridges.pipeline import pipe_stim
    from foundation.stimuli import stimulus
    from foundation.recordings import trials

    # scan key
    scan_key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx)
    scan_trials = pipe_stim.Trial & scan_key

    # stimulus types
    conditions = pipe_stim.Condition & scan_trials
    stim_types = dj.U("stimulus_type") & conditions
    stim_types = stim_types.fetch("stimulus_type")

    # fill stimulus types
    for stim_type in stim_types:

        table = stim_type.split(".")[1]
        table = getattr(stimulus, table, None)

        if table is None:
            raise NotImplementedError(f"Condition {stim_type} is not yet implemented")
        else:
            conds = conditions & dict(stimulus_type=stim_type)
            table.insert(conds.proj(), skip_duplicates=True)

    # populate stimulus
    stimulus.StimulusLink.fill()
    stimulus.Stimulus.populate(reserve_jobs=reserve_jobs, display_progress=display_progress)

    # populate trial
    trials.ScanTrial.insert(scan_trials.proj(), skip_duplicates=True)
    trials.TrialLink.fill()
    trials.Trial.populate(reserve_jobs=reserve_jobs, display_progress=display_progress)

    # populate trials
    trials.ScanTrials.insert1(scan_key, skip_duplicates=True)
    trials.TrialsLink.fill()
    trials.Trials.populate(reserve_jobs=reserve_jobs, display_progress=display_progress)


# ---------- Loading Functions ----------


def pipeline(animal_id, session, scan_idx):
    """
    Parameters
    ----------
    animal_id : int
        animal id
    session : int
        scan session
    scan_idx : int
        scan index

    Returns
    -------
    dj.schemas.VirtualModule
        pipeline_meso | pipeline_reso
    """
    from foundation.bridges.pipeline import pipe_fuse, pipe_meso, pipe_reso

    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx)

    pipe = dj.U("pipe") & (pipe_fuse.ScanDone & key)
    pipe = pipe.fetch1("pipe")

    if pipe == "meso":
        return pipe_meso

    elif pipe == "reso":
        return pipe_reso

    else:
        raise ValueError(f"{pipe} not recognized")


def planes(animal_id, session, scan_idx):
    """
    Parameters
    ----------
    animal_id : int
        animal id
    session : int
        scan session
    scan_idx : int
        scan index

    Returns
    -------
    int
        number of scan plane (z-depths)
    """
    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx)
    pipe = pipeline(**key)

    n = (pipe.ScanInfo & key).proj(n="nfields div nrois").fetch1("n")
    if n != len(dj.U("z") & (pipe.ScanInfo.Field & key)):
        raise ValueError("unexpected number of depths")

    return n


def stimulus_times(animal_id, session, scan_idx):
    """
    Parameters
    ----------
    animal_id : int
        animal id
    session : int
        scan session
    scan_idx : int
        scan index

    Returns
    -------
    1D array
        start of each scan volume on the stimulus clock
    """
    from foundation.bridges.pipeline import pipe_stim

    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx)
    n = planes(**key)
    times = (pipe_stim.Sync & key).fetch1("frame_times")[::n]
    return times


def behavior_times(animal_id, session, scan_idx):
    """
    Parameters
    ----------
    animal_id : int
        animal id
    session : int
        scan session
    scan_idx : int
        scan index

    Returns
    -------
    1D array
        start of each scan volume on the behavior clock
    """
    from foundation.bridges.pipeline import pipe_stim

    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx)
    n = planes(**key)
    times = (pipe_stim.BehaviorSync & key).fetch1("frame_times")[::n]
    return times


def eye_times(animal_id, session, scan_idx):
    """
    Parameters
    ----------
    animal_id : int
        animal id
    session : int
        scan session
    scan_idx : int
        scan index

    Returns
    -------
    1D array
        eye trace times on the stimulus clock
    """
    from foundation.bridges.pipeline import pipe_eye

    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx)

    stim_times = stimulus_times(**key)
    beh_times = behavior_times(**key)

    stim_median = np.median(stim_times)
    beh_median = np.median(beh_times)

    beh_to_stim = interp1d(
        x=beh_times - beh_median,
        y=stim_times - stim_median,
        kind="linear",
        fill_value=np.nan,
        bounds_error=False,
        copy=False,
    )

    raw = (pipe_eye.Eye & key).fetch1("eye_time")
    nans = np.isnan(raw)
    times = np.full_like(raw, np.nan)
    times[~nans] = beh_to_stim(raw[~nans] - beh_median) + stim_median

    return times


def treadmill_times(animal_id, session, scan_idx):
    """
    Parameters
    ----------
    animal_id : int
        animal id
    session : int
        scan session
    scan_idx : int
        scan index

    Returns
    -------
    1D array
        treadmill trace times on the stimulus clock
    """
    from foundation.bridges.pipeline import pipe_tread

    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx)

    stim_times = stimulus_times(**key)
    beh_times = behavior_times(**key)

    stim_median = np.median(stim_times)
    beh_median = np.median(beh_times)

    beh_to_stim = interp1d(
        x=beh_times - beh_median,
        y=stim_times - stim_median,
        kind="linear",
        fill_value=np.nan,
        bounds_error=False,
        copy=False,
    )

    raw = (pipe_tread.Treadmill & key).fetch1("treadmill_time")
    nans = np.isnan(raw)
    times = np.full_like(raw, np.nan)
    times[~nans] = beh_to_stim(raw[~nans] - beh_median) + stim_median

    return times
