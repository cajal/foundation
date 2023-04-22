import numpy as np
import datajoint as dj
from djutils import link, group, method, row_method, row_property, MissingError
from foundation.utils.logging import logger
from foundation.utils.trace import monotonic
from foundation.bridge.pipeline import pipe_stim
from foundation.stimulus import video

schema = dj.schema("foundation_recording")


# -------------- Trial --------------

# -- Trial Base --


class TrialBase:
    """Recording Trial"""

    @row_property
    def video_frames(self):
        """
        Returns
        -------
        video.VideoFrames
            stimulus tuple
        """
        raise NotImplementedError()

    @row_property
    def flips(self):
        """
        Returns
        -------
        1D array
            stimulus flip times
        """
        raise NotImplementedError()


# -- Trial Types --


@schema
class ScanTrial(TrialBase, dj.Lookup):
    definition = """
    -> pipe_stim.Trial
    """

    @row_property
    def stimulus(self):
        trial = pipe_stim.Trial * pipe_stim.Condition & self
        stim_type = trial.fetch1("stimulus_type")
        stim_type = stim_type.split(".")[1]
        return stimulus.StimulusLink.get(stim_type, trial)

    @row_property
    def flips(self):
        return (pipe_stim.Trial & self).fetch1("flip_times", squeeze=True)


# -- Trial Link --


@link(schema)
class TrialLink:
    links = [ScanTrial]
    name = "trial"
    comment = "recording trial"


# -- Computed Trial --


@schema
class TrialFlips(dj.Computed):
    definition = """
    -> TrialLink
    ---
    flips       : int unsigned      # number of stimulus flips
    flip_start  : double            # time of first flip
    flip_end    : double            # time of last flip
    """

    def make(self, key):
        flips = (TrialLink & key).link.flips

        assert np.isfinite(flips).all()
        assert monotonic(flips)

        key["flips"] = len(flips)
        key["flip_start"] = flips[0]
        key["flip_end"] = flips[-1]

        self.insert1(key)


# -------------- Trial Filter --------------

# -- Trial Filter Base --


class TrialFilterBase:
    """Trial Filter"""

    @row_method
    def filter(self, trials):
        """
        Parameters
        ----------
        trials : Trial
            Trial tuples

        Returns
        -------
        Trial
            retricted Trial tuples
        """
        raise NotImplementedError()


# -- Trial Filter Types --


@method(schema)
class FlipsEqualsFrames(TrialFilterBase):
    name = "flips_equals_frames"
    comment = "flips == frames"

    @row_method
    def filter(self, trials):
        key = (trials * stimulus.Stimulus) & "flips = frames"
        return trials & key.proj()


@schema
class StimulusType(TrialFilterBase, dj.Lookup):
    definition = """
    stimulus_type       : varchar(128)  # stimulus type
    """

    def filter(self, trials):
        return trials & (stimulus.StimulusLink & self)


# -- Trial Filter Link --


@link(schema)
class TrialFilterLink:
    links = [FlipsEqualsFrames, StimulusType]
    name = "trial_filter"
    comment = "recording trial filter"


# -- Trial Filter Group --


@group(schema)
class TrialFilterGroup:
    keys = [TrialFilterLink]
    name = "trial_filters"
    comment = "recording trial filters"
