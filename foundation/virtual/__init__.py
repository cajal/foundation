from datajoint import create_virtual_module

utility = create_virtual_module("utility", "foundation_utility", create_schema=True)
stimulus = create_virtual_module("stimulus", "foundation_stimulus", create_schema=True)
scan = create_virtual_module("scan", "foundation_scan", create_schema=True)
recording = create_virtual_module("recording", "foundation_recording", create_schema=True)
fnn = create_virtual_module("fnn", "foundation_fnn", create_schema=True)
function = create_virtual_module("function", "foundation_function", create_schema=True)
