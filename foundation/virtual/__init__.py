from datajoint import create_virtual_module

utility = create_virtual_module("utility", "foundation_utility")
stimulus = create_virtual_module("stimulus", "foundation_stimulus")
scan = create_virtual_module("scan", "foundation_scan")
recording = create_virtual_module("recording", "foundation_recording")
fnn = create_virtual_module("fnn", "foundation_fnn")
tuning = create_virtual_module("tuning", "foundation_tuning")
