import datajoint as dj
import numpy as np

from foundation.utils.traces import Sample
from foundation.utils.logging import logger


def pipe(animal_id, session, scan_idx):
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

    fuse = dj.create_virtual_module("fuse", "pipeline_fuse")
    pipe = dj.U("pipe") & (fuse.ScanDone & key)
    pipe = pipe.fetch1("pipe")

    if pipe == "meso":
        return dj.create_virtual_module("meso", "pipeline_meso")

    elif pipe == "reso":
        return dj.create_virtual_module("reso", "pipeline_meso")

    else:
        raise ValueError(f"{pipe} not recognized")


def scan_times(animal_id, session, scan_idx):
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
    pipeline = pipe(**key)

    # number of planes
    n = (pipeline.ScanInfo & key).proj(n="nfields div nrois").fetch1("n")
    if n != len(dj.U("z") & (pipeline.ScanInfo.Field & key)):
        raise ValueError("unexpected number of depths")

    # fetch times
    stimulus = dj.create_virtual_module("stimulus", "pipeline_stimulus")
    fetch = lambda sync: (sync & key).fetch1("frame_times")[::n]
    stimulus_time, behavior_time = map(fetch, [stimulus.Sync, stimulus.BehaviorSync])

    # verify times
    assert np.isfinite(stimulus_time).all()
    assert np.isfinite(behavior_time).all()

    return stimulus_time, behavior_time, pipeline


def sample_response(
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
    List[dj.key]
        unit keys
    Sample
        samples responses by the stimulus clock
    """
    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx)
    time, _, pipeline = scan_times(**key)

    # restrict units
    key = dict(
        key,
        pipe_version=pipe_version,
        segmentation_method=segmentation_method,
        spike_method=spike_method,
    )
    if unit_ids is None:
        units = pipeline.ScanSet.UnitInfo * pipeline.Activity.Trace & key
    else:
        unit_keys = [dict(key, unit_id=unit_id) for unit_id in unit_ids]
        units = pipeline.ScanSet.UnitInfo * pipeline.Activity.Trace & unit_keys

    # fetch traces and offsets
    keys, ms_delays, traces = units.fetch(dj.key, "ms_delay", "trace", order_by=units.primary_key)
    offsets = ms_delays / 1000

    # verify number of tuples
    if unit_ids is not None:
        assert len(keys) == len(unit_ids)

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
