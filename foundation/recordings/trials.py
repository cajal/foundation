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
        str
            primary key of stimulus.Stimulus
        int
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
            logger.warning(f"Skipping {key} because stimulus not found.")
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


# ---------- Trial Link ----------


@link(schema)
class Trial:
    links = [ScanTrial]
    name = "trial"
    comment = "recording trial"


# ---------- Trials Base ----------


class TrialsBase:
    _definition = """
    {key}
    ---
    trials              : int unsigned  # number of trials
    """

    class Trial(dj.Part):
        definition = """
        -> master
        -> Trial
        """

    @staticmethod
    def _trials(**key):
        """
        Parameters
        ----------
        key : dict
            primary key

        Returns
        -------
        Trial
            restricted Trial table

        Raises
        ------
        MissingError
            if trial is missing
        """
        raise NotImplementedError()

    def make(self, key):
        try:
            trials = self._trials(**key)

        except MissingError:
            logger.warning(f"Skipping {key} because trial not found.")
            return

        assert isinstance(trials, Trial), f"Expected Trial table but got {trials.__class__}"

        master_key = dict(key, trials=len(trials))
        self.insert1(master_key)

        part_keys = (self & key).proj() * trials.proj()
        self.Trial.insert(part_keys)


# ---------- Trials Types ----------


pipeline_experiment = dj.create_virtual_module("pipeline_experiment", "pipeline_experiment")


@schema
class ScanTrials(TrialsBase, dj.Computed):
    definition = TrialsBase._definition.format(key="-> pipeline_experiment.Scan")

    @staticmethod
    def _trials(**key):
        all_trials = pipeline_stimulus.Trial & key
        trials = Trial & (Trial.ScanTrial * ScanTrial & key)

        if len(trials) == len(all_trials):
            return trials
        else:
            raise MissingError()


# ---------- Trials Link ----------


@link(schema)
class Trials:
    links = [ScanTrials]
    name = "trials"
    comment = "recording trials"
