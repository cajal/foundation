import datajoint as dj

pipe_stim = dj.create_virtual_module("pipe_stim", "pipeline_stimulus")
pipe_fuse = dj.create_virtual_module("pipe_fuse", "pipeline_fuse")
pipe_meso = dj.create_virtual_module("pipe_meso", "pipeline_meso")
pipe_reso = dj.create_virtual_module("pipe_reso", "pipeline_reso")
pipe_eye = dj.create_virtual_module("pipe_eye", "pipeline_eye")
pipe_tread = dj.create_virtual_module("pipe_tread", "pipeline_treadmill")