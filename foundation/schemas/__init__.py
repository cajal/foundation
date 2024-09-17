from datajoint import config
from djutils import Schema

config["stores"] = {
    "external": dict(
        protocol="file",
        location="/mnt/scratch09/scratch09/foundation/external/",
    ),
}

utility = Schema("foundation_utility")
stimulus = Schema("foundation_stimulus")
scan = Schema("foundation_scan")
recording = Schema("foundation_recording")
fnn = Schema("foundation_fnn")
tuning = Schema("foundation_tuning")
