import numpy as np
import datajoint as dj
from foundation.stimuli import stimulus
from foundation.utils.traces import Sample, fill_nans
from foundation.utils.splines import spline
from foundation.utils.logging import logger

stim = dj.create_virtual_module("stim", "pipeline_stimulus")
fuse = dj.create_virtual_module("fuse", "pipeline_fuse")
meso = dj.create_virtual_module("meso", "pipeline_meso")
reso = dj.create_virtual_module("reso", "pipeline_reso")
pupil = dj.create_virtual_module("pupil", "pipeline_eye")
tread = dj.create_virtual_module("tread", "pipeline_treadmill")


# ---------- Populate Functions ----------


def populate_stimuli(animal_id, session, scan_idx, reserve_jobs=True, display_progress=True):
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
    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx)

    # scan trials
    trials = stim.Trial & key

    # stimulus types
    stim_types = dj.U("stimulus_type") & (stim.Condition & trials)
    stim_types = stim_types.fetch("stimulus_type")

    # populate each stimulus table
    for stim_type in stim_types:

        table = stim_type.split(".")[1]
        table = getattr(stimulus, table)

        if table is None:
            raise NotImplementedError(f"Condition {stim_type} is not yet implemented")
        else:
            table.populate(trials, reserve_jobs=reserve_jobs, display_progress=display_progress)

    # fill links
    stimulus.Stimulus.fill()


# ---------- Loading Functions ----------


def load_pipe(animal_id, session, scan_idx):
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
    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx)

    pipe = dj.U("pipe") & (fuse.ScanDone & key)
    pipe = pipe.fetch1("pipe")

    if pipe == "meso":
        return meso

    elif pipe == "reso":
        return reso

    else:
        raise ValueError(f"{pipe} not recognized")


def load_scan_times(animal_id, session, scan_idx):
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
        start of each scan volume in stimulus clock
    1D array
        start of each scan volume in behavior clock
    dj.schemas.VirtualModule
        pipeline_meso | pipeline_reso
    """
    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx)
    pipe = load_pipe(**key)

    # number of planes
    n = (pipe.ScanInfo & key).proj(n="nfields div nrois").fetch1("n")
    if n != len(dj.U("z") & (pipe.ScanInfo.Field & key)):
        raise ValueError("unexpected number of depths")

    # fetch times
    fetch = lambda sync: (sync & key).fetch1("frame_times")[::n]
    stimulus_time, behavior_time = map(fetch, [stim.Sync, stim.BehaviorSync])

    # verify times
    assert np.isfinite(stimulus_time).all()
    assert np.isfinite(behavior_time).all()

    return stimulus_time, behavior_time, pipe


def load_response_sampler(
    animal_id,
    session,
    scan_idx,
    pipe_version=1,
    segmentation_method=6,
    spike_method=6,
    unit_ids=None,
    target_period=0.1,
    tolerance=1,
    kind="hamming",
    **kwargs,
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
    pipe_version : int
        pipeline version
    segmentation_method : int
        unit segmentation method
    spike_method : int
        unit spike method
    unit_ids : Sequence[int] | None
        unit ids -- None returns all units in scan
    target_period : float
        target sampling period
    tolerace : int
        tolerance for time and response length mismatches
    kind : str
        specifies the kind of sampling
    kwargs : dict
        additional sampling options

    Returns
    -------
    List[dict]
        unit keys
    foundation.utils.traces.Sample
        samples responses by the stimulus clock
    """
    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx)
    time, _, pipe = load_scan_times(**key)

    # restrict units
    key = dict(
        key,
        pipe_version=pipe_version,
        segmentation_method=segmentation_method,
        spike_method=spike_method,
    )
    if unit_ids is None:
        units = pipe.ScanSet.UnitInfo * pipe.Activity.Trace & key
        n = len(units)
    else:
        unit_keys = [dict(key, unit_id=unit_id) for unit_id in unit_ids]
        units = pipe.ScanSet.UnitInfo * pipe.Activity.Trace & unit_keys
        n = len(unit_keys)

    # fetch traces and offsets
    logger.info(f"Fetching {n} response traces")
    keys, ms_delays, traces = units.fetch(dj.key, "ms_delay", "trace", order_by=units.primary_key)
    offsets = ms_delays / 1000

    # verify number of tuples
    assert len(keys) == n, f"Expected {n} traces but found {len(keys)}"

    # response sampler
    sample = Sample(
        time=time,
        traces=traces,
        target_period=target_period,
        offsets=offsets,
        tolerance=tolerance,
        kind=kind,
        **kwargs,
    )

    return keys, sample


def load_pupil_sampler(
    animal_id,
    session,
    scan_idx,
    tracking_method=2,
    target_period=0.1,
    tolerance=1,
    kind="hamming",
    **kwargs,
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
    tracking_method : int
        pupil tracking method
    target_period : float
        target sampling period
    tolerace : int
        tolerance for time and response length mismatches
    kind : str
        specifies the kind of sampling
    kwargs : dict
        additional sampling options

    Returns
    -------
    scipy.interpolate.InterpolatedUnivariateSpline
        converts stimulus to behavior clock
    foundation.utils.traces.Sample
        samples pupil traces (Radius, Center X, Center Y) by the behavior clock
    """
    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx)

    # fetch times
    eye_time = (pupil.Eye & key).fetch1("eye_time", squeeze=True)

    # fill NaN treadmill times
    if np.isnan(eye_time).any():
        logger.info("Replacing NaNs in eye time with interpolated values")
        eye_time = fill_nans(eye_time)

    # verify tracking exists
    fit_key = pupil.FittedPupil & dict(key, tracking_method=tracking_method)
    fit_key = fit_key.fetch1(dj.key)

    # fetch pupil fits
    logger.info("Fetching pupil traces")
    fits = pupil.FittedPupil.Circle & fit_key
    radius, center = fits.fetch("radius", "center", order_by="frame_id")

    # deal with NaNs
    finite = np.isfinite(radius)
    xy = np.full([len(radius), 2], np.nan)
    xy[finite] = np.stack(center[finite])
    x, y = xy.T

    # traces: Radius, Center X, Center Y
    traces = [radius, x, y]

    # pupil sampler
    sample = Sample(
        time=eye_time,
        traces=traces,
        target_period=target_period,
        tolerance=tolerance,
        kind=kind,
        **kwargs,
    )

    return sample


def load_treadmill_sampler(
    animal_id,
    session,
    scan_idx,
    target_period=0.1,
    tolerance=1,
    kind="hamming",
    **kwargs,
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
    target_period : float
        target sampling period
    tolerace : int
        tolerance for time and response length mismatches
    kind : str
        specifies the kind of sampling
    kwargs : dict
        additional sampling options

    Returns
    -------
    scipy.interpolate.InterpolatedUnivariateSpline
        converts stimulus to behavior clock
    foundation.utils.traces.Sample
        samples treadmill trace (velocity) by the behavior clock
    """
    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx)

    # fetch times and velocities
    tread_time, tread_vel = (tread.Treadmill & key).fetch1("treadmill_time", "treadmill_vel", squeeze=True)

    # fill NaN treadmill times
    if np.isnan(tread_time).any():
        logger.info("Replacing NaNs in eye time with interpolated values")
        tread_time = fill_nans(tread_time)

    # treadmill sampler
    sample = Sample(
        time=tread_time,
        traces=[tread_vel],
        target_period=target_period,
        tolerance=tolerance,
        kind=kind,
        **kwargs,
    )

    return sample
