import datajoint as dj

pipe_exp = dj.create_virtual_module("pipe_exp", "pipeline_experiment")
pipe_shared = dj.create_virtual_module("pipe_shared", "pipeline_shared")
pipe_stim = dj.create_virtual_module("pipe_stim", "pipeline_stimulus")
pipe_fuse = dj.create_virtual_module("pipe_fuse", "pipeline_fuse")
pipe_meso = dj.create_virtual_module("pipe_meso", "pipeline_meso")
pipe_reso = dj.create_virtual_module("pipe_reso", "pipeline_reso")
pipe_eye = dj.create_virtual_module("pipe_eye", "pipeline_eye")
pipe_tread = dj.create_virtual_module("pipe_tread", "pipeline_treadmill")


def resolve_pipe(animal_id, session, scan_idx):
    """
    Parameters
    ----------
    animal_id : int
        animal id
    session : int
        session
    scan_idx : int
        scan idx

    Returns
    -------
        pipe_meso | pipe_reso
    """
    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx)

    pipe = dj.U("pipe") & (pipe_fuse.ScanDone & key)
    pipe = pipe.fetch1("pipe")

    if pipe == "meso":
        return pipe_meso
    elif pipe == "reso":
        return pipe_reso
    else:
        raise ValueError(f"{pipe} not recognized")
