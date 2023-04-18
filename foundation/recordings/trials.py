import numpy as np
import datajoint as dj
from djutils import link, group, method, row_method, row_property, MissingError
from foundation.stimuli import stimulus
from foundation.utils.logging import logger

pipe_stim = dj.create_virtual_module("pipe_stim", "pipeline_stimulus")
pipe_exp = dj.create_virtual_module("pipe_exp", "pipeline_experiment")
schema = dj.schema("foundation_recordings")


# -------------- Trial --------------

# -- Base --


class TrialBase:
    @property
    def stimulus(self):
        """
        Returns
        -------
        stimulus.Stimulus
            stimulus tuple

        Raises
        ------
        MissingError
            if stimulus is missing
        """
        raise NotImplementedError()

    @property
    def flips(self):
        """
        Returns
        -------
        1D array
            stimulus flip times

        Raises
        ------
        MissingError
            if flip times are missing
        """
        raise NotImplementedError()


# -- Types --


@schema
class ScanTrial(TrialBase, dj.Lookup):
    definition = """
    -> pipe_stim.Trial
    """

    @property
    def stimulus(self):
        trial = pipe_stim.Trial * pipe_stim.Condition & self
        stim_type = trial.fetch1("stimulus_type")
        stim_type = stim_type.split(".")[1]
        return stimulus.StimulusLink.get(stim_type, trial)

    @property
    def flips(self):
        return (pipe_stim.Trial & self).fetch1("flip_times", squeeze=True)


# -- Link --


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
    flips                   : int unsigned      # number of stimulus flips
    """

    def make(self, key):
        link = (TrialLink & key).link

        try:
            stimulus = link.stimulus

        except MissingError:
            logger.warning(f"Missing stimulus. Skipping {key}.")
            return

        try:
            flips = link.flips

        except MissingError:
            logger.warning(f"Missing stimulus. Skipping {key}.")
            return

        key["stimulus_id"] = stimulus.fetch1("stimulus_id")
        key["flips"] = len(flips)
        self.insert1(key)


# -------------- Trials --------------

# -- Base --


class TrialsBase:
    @property
    def trials(self):
        """
        Returns
        -------
        Trial
            restricted Trial table

        Raises
        ------
        MissingError
            if trials are missing
        """
        raise NotImplementedError()


# -- Types --


@schema
class ScanTrials(TrialsBase, dj.Lookup):
    definition = """
    -> pipe_exp.Scan
    """

    @property
    def trials(self):
        all_trials = pipe_stim.Trial & self
        trials = Trial & (TrialLink.ScanTrial * ScanTrial & self)

        if all_trials - trials:
            raise MissingError()

        return trials


# -- Link --


@link(schema)
class TrialsLink:
    links = [ScanTrials]
    name = "trials"
    comment = "recording trials"


class ComputedTrialsBase:
    @property
    def trials(self):
        key, n = self.fetch1(dj.key, "trials")
        trials = self.Trial & key

        if len(trials) == n:
            return trials
        else:
            raise MissingError("Trials are missing.")


@schema
class Trials(ComputedTrialsBase, dj.Computed):
    definition = """
    -> TrialsLink
    ---
    trials              : int unsigned      # number of trials
    """

    class Trial(dj.Part):
        definition = """
        -> master
        -> Trial
        """

    def make(self, key):
        link = (TrialsLink & key).link

        try:
            trials = link.trials

        except MissingError:
            logger.warning(f"Mising trials. Skipping {key}.")
            return

        master_key = dict(key, trials=len(trials))
        self.insert1(master_key)

        part_keys = (self & key).proj() * trials.proj()
        self.Trial.insert(part_keys)

    @property
    def trials(self):
        key, n = self.fetch1(dj.key, "trials")
        trials = self.Trial & key

        if len(trials) == n:
            return trials
        else:
            raise MissingError("Trials are missing.")


# -------------- Trial Filter --------------

# -- Base --


class TrialFilterBase:
    """Trial Filter"""

    @row_method
    def filter(self, trials):
        """
        Parameters
        ----------
        trials : Trial
            tuples from Trial

        Returns
        -------
        Trial
            retricted tuples from Trial
        """
        raise NotImplementedError()


# -- Types --


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


# -- Link --


@link(schema)
class TrialFilterLink:
    links = [FlipsEqualsFrames, StimulusType]
    name = "trial_filter"
    comment = "recording trial filter"


# -- Group --


@group(schema)
class TrialFilters:
    keys = [TrialFilterLink]
    name = "trial_filters"
    comment = "recording trial filters"


# -------------- Filtered Trials --------------


@schema
class FilteredTrials(ComputedTrialsBase, dj.Computed):
    definition = """
    -> Trials
    -> TrialFilters
    ---
    trials              : int unsigned      # number of trials
    """

    class Trial(dj.Part):
        definition = """
        -> master
        -> Trial
        """

    def make(self, key):
        trials = (Trials & key).trials
        trials = Trial & trials

        filters = (TrialFilters & key).members
        filters = filters.fetch(dj.key, order_by=filters.primary_key)

        keys = [(TrialFilterLink & filt).link.filter(trials).proj() for filt in filters]
        filtered = trials & dj.AndList(keys)

        master_key = dict(key, trials=len(filtered))
        self.insert1(master_key)

        part_key = (self & master_key).proj() * filtered.proj()
        self.Trial.insert(part_key)
