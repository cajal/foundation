from datajoint import config
from djutils import Schema

config["stores"] = {
    "scratch09": dict(
        protocol="file",
        location="/mnt/scratch09/foundation/",
        stage="/mnt/scratch09/foundation/",
    )
}

utility = Schema("foundation_utility")
stimulus = Schema("foundation_stimulus")
recording = Schema("foundation_recording")
scan = Schema("foundation_scan")
