from djutils import keys, U
from foundation.virtual.bridge import pipe_exp, pipe_stim


@keys
class VisualScanVideo:
    """Visual Scan Video"""

    @property
    def keys(self):
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
        keys = [Video.query(_, trials).proj() for _ in link_types]
        VideoInfo.populate(keys, reserve_jobs=True, display_progress=True)


@keys
class VisualScanFrameList:
    """Visual scan with static images presented"""

    @property
    def keys(self):
        return [
            pipe_exp.Scan & (
                pipe_stim.Trial * pipe_stim.Condition & pipe_stim.Frame
            ),
        ]
    
    def fill(self):
        from foundation.stimulus.video import FrameList

        keys = U('condition_hash').aggr(
            (pipe_stim.Frame * pipe_stim.Condition * pipe_stim.Trial & self.key),
            trial_idx='MIN(trial_idx)'
        ).fetch(
            "KEY", order_by="trial_idx ASC"
        )
        return FrameList.fill(
            restrictions=keys,
            note=(
                "All unique stimulus.Frame conditions presented in "\
                f"{self.item['animal_id']}-{self.item['session']}-{self.item['scan_idx']}, "\
                "ordered by the trial_idx of the first repetition.",
            ),
        )