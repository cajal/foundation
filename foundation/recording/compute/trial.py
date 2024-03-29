import numpy as np
from djutils import keys, rowproperty
from foundation.virtual.bridge import pipe_stim
from foundation.virtual import utility, recording


# ----------------------------- Trial -----------------------------

# -- Trial Interface --


class TrialType:
    """Recording Trial"""

    @rowproperty
    def bounds(self):
        """
        Returns
        -------
        float
            trial start time (seconds)
        float
            trial end time (seconds)
        """
        raise NotImplementedError()

    @rowproperty
    def video_id(self):
        """
        Returns
        -------
        str
            key (foundation.stimulus.video.Video)
        """
        raise NotImplementedError()

    @rowproperty
    def flip_times(self):
        """
        Returns
        -------
        1D array
            flip times (seconds) of stimulus frames
        """
        raise NotImplementedError()


# -- Trial Types --


@keys
class ScanTrial(TrialType):
    """Scan Trial"""

    @property
    def keys(self):
        return [
            pipe_stim.Trial,
        ]

    @rowproperty
    def bounds(self):
        from foundation.utils.resample import monotonic

        flip_times = self.flip_times
        assert np.isfinite(flip_times).all()
        assert monotonic(flip_times)
        return flip_times[0], flip_times[-1]

    @rowproperty
    def video_id(self):
        from foundation.stimulus.video import Video

        trial = pipe_stim.Trial * pipe_stim.Condition & self.item
        stim_type = trial.fetch1("stimulus_type")
        stim_type = stim_type.split(".")[1]
        return Video.query(stim_type, trial).fetch1("video_id")

    @rowproperty
    def flip_times(self):
        return (pipe_stim.Trial & self.item).fetch1("flip_times", squeeze=True)

