import numpy as np
import datajoint as dj
from djutils import link
from foundation.stimuli import stimulus
from foundation.utils.errors import MissingError
from foundation.utils.logging import logger

schema = dj.schema("foundation_recordings")


# ---------- Trial Base ----------


class TrialBase:
    _definition = """
    {key}
    ---
    -> stimulus.Stimulus
    frames_match            : bool      # number of frames match
    """

    @staticmethod
    def _stimulus_frames(**key):
        """
        Parameters
        ----------
        key : dict
            primary key

        Returns
        -------
        stimulus_id : str
            primary key of stimulus.Stimulus
        frames : int
            number of frames

        Raises
        ------
        MissingError
            if stimulus is missing
        """
        raise NotImplementedError()

    def make(self, key):
        try:
            stimulus_id, trial_frames = self._stimulus_frames(**key)

        except MissingError:
            logger.warning("Skipping trial because stimulus not found. Populate stimuli before populating trials.")
            return

        if "stimulus_id" in key:
            assert stimulus_id == key["stimulus_id"]
        else:
            key["stimulus_id"] = str(stimulus_id)

        stimulus_frames = (stimulus.Stimulus & key).link.fetch1("frames")
        key["frames_match"] = bool(trial_frames == stimulus_frames)

        self.insert1(key)


# ---------- Trial Types ----------

pipeline_stimulus = dj.create_virtual_module("pipeline_stimulus", "pipeline_stimulus")


@schema
class ScanTrial(TrialBase, dj.Computed):
    definition = TrialBase._definition.format(key="-> pipeline_stimulus.Trial")

    @staticmethod
    def _stimulus_frames(**key):
        trial = pipeline_stimulus.Trial * pipeline_stimulus.Condition & key

        stim_type, flip_times = trial.fetch1("stimulus_type", "flip_times", squeeze=True)
        frames = len(flip_times)

        stim_type = stim_type.split(".")[1]
        stim = stimulus.Stimulus.join(stim_type, trial)

        if stim is None:
            raise MissingError()
        else:
            stimulus_id = stim.fetch1("stimulus_id")

        return stimulus_id, frames
