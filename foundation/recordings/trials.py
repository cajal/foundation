import numpy as np
import datajoint as dj
from djutils import link, group, method, row_method, row_property, MissingError
from foundation.utils.logging import logger
from foundation.stimuli import stimulus

pipe_stim = dj.create_virtual_module("pipe_stim", "pipeline_stimulus")
pipe_exp = dj.create_virtual_module("pipe_exp", "pipeline_experiment")
schema = dj.schema("foundation_recordings")


# -------------- Trial --------------

# -- Trial Base --


class TrialBase:
    """Recording Trial"""

    @row_property
    def stimulus(self):
        """
        Returns
        -------
        stimulus.Stimulus
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
class Trial(TrialBase, dj.Computed):
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
            logger.warning(f"Missing flips. Skipping {key}.")
            return

        key["stimulus_id"] = stimulus.fetch1("stimulus_id")
        key["flips"] = len(flips)
        self.insert1(key)

    @row_property
    def stimulus(self):
        return stimulus.Stimulus & self

    @row_property
    def flips(self):
        flips = (TrialLink & self).link.flips

        if len(flips) != self.fetch1("flips"):
            raise ValueError("Flip numbers do not match.")

        if not np.all(np.diff(flips) > 0):
            raise ValueError("Flips do not monotonically increase in time.")

        return flips


# -------------- Trials --------------

# -- Trials Base --


class TrialsBase:
    """Recording Trials"""

    @row_property
    def trials(self):
        """
        Returns
        -------
        Trial
            Trial tuples
        """
        raise NotImplementedError()


# -- Trials Types --


@schema
class ScanTrials(TrialsBase, dj.Lookup):
    definition = """
    -> pipe_exp.Scan
    """

    @row_property
    def trials(self):
        all_trials = pipe_stim.Trial & self
        trials = Trial & (TrialLink.ScanTrial * ScanTrial & self)

        if all_trials - trials:
            raise MissingError()

        return trials


# -- Trials Link --


@link(schema)
class TrialsLink:
    links = [ScanTrials]
    name = "trials"
    comment = "recording trials"


# -- Computed Trials --


class ComputedTrialsBase(TrialsBase):
    @row_property
    def trials(self):
        key, n = self.fetch1(dj.key, "trials")
        trials = Trial & (self.Trial & key)

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
class TrialFilters:
    keys = [TrialFilterLink]
    name = "trial_filters"
    comment = "recording trial filters"


# -- Computed Trial Filter --


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
        filters = (TrialFilters & key).members

        keys = filters.fetch(dj.key, order_by=filters.primary_key)
        keys = [(TrialFilterLink & k).link.filter(trials).proj() for k in keys]

        filtered = trials & dj.AndList(keys)

        master_key = dict(key, trials=len(filtered))
        self.insert1(master_key)

        part_key = (self & master_key).proj() * filtered.proj()
        self.Trial.insert(part_key)
