import datajoint as dj

dj.config["stores"] = {
    "scratch09": dict(
        protocol="file",
        location="/mnt/scratch09/foundation/",
        stage="/mnt/scratch09/foundation/",
    )
}

utility = dj.schema("foundation_utility")
stimulus = dj.schema("foundation_stimulus")
recording = dj.schema("foundation_recording")
scan = dj.schema("foundation_scan")
