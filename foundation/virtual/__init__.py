import datajoint as dj

stimulus = dj.create_virtual_module("stimulus", "foundation_stimulus")
scan = dj.create_virtual_module("scan", "foundation_scan")
recording = dj.create_virtual_module("recording", "foundation_recording")
