import datajoint as dj
import numpy as np

from foundation.utils.logging import logger
from foundation.utils.splines import spline


def pipe(animal_id, session, scan_idx):
    """
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


def scan_times(animal_id, session, scan_idx):
    """
    Parameters
    ----------
    animal_id : int
    session : int
    scan_idx : int

    Returns
    -------
    np.array
        start of each scan volume in stimulus clock
    np.array
        start of each scan volume in behavior clock
    """
    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx)

    info = pipe(**key).ScanInfo
    n = (info & key).proj(n="nfields div nrois").fetch1("n")
    if n != len(dj.U("z") & (info.Field & key)):
        raise ValueError("unexpected number of depths")

    stimulus = dj.create_virtual_module("stimulus", "pipeline_stimulus")
    fetch = lambda sync: (sync & key).fetch1("frame_times")[::n]
    stimulus_times, behavior_times = map(fetch, [stimulus.Sync, stimulus.BehaviorSync])

    return stimulus_times, behavior_times
