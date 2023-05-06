from datajoint import U
from djutils import keys
from foundation.virtual.bridge import pipe_exp, pipe_stim
from foundation.virtual import utility
from foundation.stimulus.video import VideoLink, VideoInfo
from foundation.stimulus.cache import ResizedVideo


@keys
class Scan:
    """Scan stimulus"""

    @property
    def key_list(self):
        return [
            pipe_exp.Scan,
        ]

    def fill(self):
        # scan trials
        trials = pipe_stim.Trial * pipe_stim.Condition & self.key

        # stimulus types
        stim_types = U("stimulus_type") & trials
        stim_keys = dict()

        for stim_type in stim_types.fetch("stimulus_type"):

            keys = trials & dict(stimulus_type=stim_type)
            stype = stim_type.split(".")[1]

            table = getattr(VideoLink, stype)._link
            table.insert(keys.proj(), skip_duplicates=True, ignore_extra_fields=True)

            stim_keys[stype] = keys

        # video links
        VideoLink.fill()

        # compute video
        keys = [VideoLink.get(k, v).proj() for k, v in stim_keys.items()]
        VideoInfo.populate(keys, reserve_jobs=True, display_progress=True)


@keys
class ScanCache:
    """Scan stimulus cache"""

    @property
    def key_list(self):
        return [
            pipe_exp.Scan,
            utility.ResizeLink,
            utility.Resolution,
        ]

    def fill(self):
        ResizedVideo.populate(self.key, reserve_jobs=True, display_progress=True)
