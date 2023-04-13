import datajoint as dj
import numpy as np

from foundation.utils.logging import logger
from foundation.utils.splines import spline


def load_pipe(animal_id, session, scan_idx):
    """Load pipeline virtual module

    Parameters
    ----------
    animal_id : int
    session : int
    scan_idx : int

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


def load_times(animal_id, session, scan_idx):
    """Load synchronized stimulus and behavior times

    Parameters
    ----------
    animal_id : int
    session : int
    scan_idx : int

    Returns
    -------
    np.array
        stimulus times per scan depth
    np.array
        behavior times per scan depth
    """
    pipe = load_pipe(animal_id, session, scan_idx)
    depths = len(dj.U("z") & (pipe.ScanInfo.Field & key))

    if depths != (pipe.ScanInfo & key).proj(n="nfields div nrois").fetch1("n"):
        raise ValueError("unexpected number of depths")

    stimulus = dj.create_virtual_module("stimulus", "pipeline_stimulus")

    fetch = lambda sync: (sync & key).fetch1("frame_times", squeeze=True)[::depths]
    stim_times, beh_times = map(fetch, [stimulus.Sync, stimulus.BehaviorSync])

    # if not np.isfinite(stim_times).all():
    #     raise ValueError("non-finite value in stimulus frame_times")

    # if not np.isfinite(beh_times).all():
    #     raise ValueError("non-finite value in behavior frame_times")

    # n_stim = len(stim_times)
    # n_beh = len(beh_times)

    # if abs(n_stim - n_beh) > 1:
    #     raise ValueError("stimulus and behavior times differ in length by more than 1")

    # if n_stim > n_beh:
    #     logger.info(f"truncating stimulus times by {n_stim - n_beh}")
    #     stim_times = stim_times[:n_beh]

    # elif n_beh > n_stim:
    #     logger.info(f"truncating behavior times by {n_beh - n_stim}")
    #     beh_times = beh_times[:n_stim]

    return stim_times, beh_times
