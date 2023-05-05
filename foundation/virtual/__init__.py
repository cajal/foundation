import datajoint as dj

stimulus = dj.create_virtual_module("stimulus", "foundation_stimulus", create_schema=True)
scan = dj.create_virtual_module("scan", "foundation_scan", create_schema=True)
recording = dj.create_virtual_module("recording", "foundation_recording", create_schema=True)
