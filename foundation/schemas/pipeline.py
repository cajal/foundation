import datajoint as dj

pipe_exp = dj.create_virtual_module("pipe_exp", "pipeline_experiment")
pipe_shared = dj.create_virtual_module("pipe_shared", "pipeline_shared")
pipe_stim = dj.create_virtual_module("pipe_stim", "pipeline_stimulus")
pipe_fuse = dj.create_virtual_module("pipe_fuse", "pipeline_fuse")
pipe_meso = dj.create_virtual_module("pipe_meso", "pipeline_meso")
pipe_reso = dj.create_virtual_module("pipe_reso", "pipeline_reso")
pipe_eye = dj.create_virtual_module("pipe_eye", "pipeline_eye")
pipe_tread = dj.create_virtual_module("pipe_tread", "pipeline_treadmill")
pipe_gabor = dj.create_virtual_module("pipe_gabor", "pipeline_gabor")
pipe_dot = dj.create_virtual_module("pipe_dot", "pipeline_dot")
pipe_rdk = dj.create_virtual_module("pipe_rdk", "pipeline_rdk")


def resolve_pipe(key):
    """
    Parameters
    ----------
    key
        restriction for pipe_fuse.ScanDone

    Returns
    -------
        pipe_meso | pipe_reso
    """
    pipe = dj.U("pipe") & (pipe_fuse.ScanDone & key)
    pipe = pipe.fetch1("pipe")

    if pipe == "meso":
        return pipe_meso
    elif pipe == "reso":
        return pipe_reso
    else:
        raise ValueError(f"{pipe} not recognized")
