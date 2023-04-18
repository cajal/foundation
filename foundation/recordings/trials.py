import numpy as np
import datajoint as dj
from djutils import link
from foundation.stimuli import stimulus
from foundation.utils.errors import MissingError
from foundation.utils.logging import logger

pipe_stim = dj.create_virtual_module("pipe_stim", "pipeline_stimulus")
schema = dj.schema("foundation_recordings")


# ---------- Trial Link Base ----------


class TrialBase:
    @property
    def stimulus_id(self):
        """
        Returns
        -------
        str
            primary key of stimulus.Stimulus

        Raises
        ------
        MissingError
            if stimulus is missing
        """
        raise NotImplementedError()

    @property
    def frames(self):
        """
        Returns
        -------
        int
            number of trial frames

        Raises
        ------
        MissingError
            if trial is missing
        """
        raise NotImplementedError()


# ---------- Trial Link Types ----------


@schema
class ScanTrial(TrialBase, dj.Lookup):
    definition = """
    -> pipe_stim.Trial
    """

    @property
    def stimulus_id(self):
        trial = pipe_stim.Trial * pipe_stim.Condition & self

        stim_type = trial.fetch1("stimulus_type")
        stim_type = stim_type.split(".")[1]

        stim = stimulus.StimulusLink.get(stim_type, trial)

        if stim is None:
            raise MissingError()
        else:
            return stim.fetch1("stimulus_id")

    @property
    def frames(self):
        flip_times = (pipe_stim.Trial & self).fetch1("flip_times", squeeze=True)
        return len(flip_times)


# ---------- Trial Link ----------


@link(schema)
class TrialLink:
    links = [ScanTrial]
    name = "trial"
    comment = "recording trial"


@schema
class Trial(dj.Computed):
    definition = """
    -> TrialLink
    ---
    -> stimulus.Stimulus
    frames                  : int unsigned      # number of frames
    """

    def make(self, key):
        try:
            link = (TrialLink & key).link
            stimulus_id = link.stimulus_id
            frames = link.frames

        except MissingError:
            logger.warning(f"Skipping {key} due to missing link data.")

        key["stimulus_id"] = stimulus_id
        key["frames"] = frames
        self.insert1(key)


# # ---------- Trials Base ----------


# class TrialsBase:
#     _definition = """
#     {key}
#     ---
#     trials              : int unsigned  # number of trials
#     """

#     class Trial(dj.Part):
#         definition = """
#         -> master
#         -> Trial
#         """

#     @staticmethod
#     def _trials(**key):
#         """
#         Parameters
#         ----------
#         key : dict
#             primary key

#         Returns
#         -------
#         Trial
#             restricted Trial table

#         Raises
#         ------
#         MissingError
#             if trial is missing
#         """
#         raise NotImplementedError()

#     def make(self, key):
#         try:
#             trials = self._trials(**key)

#         except MissingError:
#             logger.warning(f"Skipping {key} because trial not found.")
#             return

#         assert isinstance(trials, Trial), f"Expected Trial table but got {trials.__class__}"

#         master_key = dict(key, trials=len(trials))
#         self.insert1(master_key)

#         part_keys = (self & key).proj() * trials.proj()
#         self.Trial.insert(part_keys)


# # ---------- Trials Types ----------


# pipeline_experiment = dj.create_virtual_module("pipeline_experiment", "pipeline_experiment")


# @schema
# class ScanTrials(TrialsBase, dj.Computed):
#     definition = TrialsBase._definition.format(key="-> pipeline_experiment.Scan")

#     @staticmethod
#     def _trials(**key):
#         all_trials = pipe_stim.Trial & key
#         trials = Trial & (Trial.ScanTrial * ScanTrial & key)

#         if len(trials) == len(all_trials):
#             return trials
#         else:
#             raise MissingError()


# # ---------- Trials Link ----------


# @link(schema)
# class Trials:
#     links = [ScanTrials]
#     name = "trials"
#     comment = "recording trials"


# ---------- Trial Restriction Base ----------


# class TrialRestrictionBase:
#     @property
#     def _restrict(self):
#         """
#         Returns
#         -------
#         Callable[[Trial], Trial]
#         """
#         raise NotImplementedError()

#     @property
#     def restrict(self):
#         f = self._restrict

#         def restrict(tuples):
#             """
#             Parameters
#             ----------
#             tuples : TrialBase
#                 tuples from TrialBase

#             Returns
#             -------
#             TrialBase
#                 retricted tuples from TrialBase
#             """
#             return f(tuples)

#         return restrict
