from datajoint import U
from djutils import keys
from foundation.virtual.bridge import pipe_exp, pipe_stim
from foundation.virtual import utility


@keys
class Scan:
    """Scan stimulus"""

    @property
    def key_list(self):
        return [
            pipe_exp.Scan,
        ]

    def fill(self):
        from foundation.stimulus.video import Video, VideoInfo

        # scan trials
        trials = pipe_stim.Trial * pipe_stim.Condition & self.key

        # stimulus types
        stim_types = U("stimulus_type") & trials
        link_types = []

        for stim_type in stim_types.fetch("stimulus_type"):

            keys = trials & dict(stimulus_type=stim_type)
            stype = stim_type.split(".")[1]
            link_types.append(stype)

            table = getattr(Video, stype)._link
            table.insert(keys.proj() - table, skip_duplicates=True, ignore_extra_fields=True)

        # video links
        Video.fill()

        # compute video
        keys = [Video.get(_, trials).proj() for _ in link_types]
        VideoInfo.populate(keys, reserve_jobs=True, display_progress=True)
