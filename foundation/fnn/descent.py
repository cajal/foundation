from djutils import rowproperty
from foundation.virtual import utility, stimulus, recording
from foundation.fnn.shared import Bound
from foundation.fnn.train import Optimizer, Scheduler
from foundation.fnn.model import NetworkModel
from foundation.schemas import fnn as schema


# ----------------------------- Stimulus -----------------------------

# -- Stimulus Interface --


class StimulusType:
    """Stimulus"""

    @rowproperty
    def visual(self):
        """
        Returns
        -------
        fnn.model.stimuli.VisualStimulus
            visual stimulus
        """
        raise NotImplementedError()


# -- Stimulus Types --


@schema.lookup
class VisualNlm(StimulusType):
    definition = """
    -> Bound
    spatial_std     : decimal(6, 4)     # spatial standard deviation
    temporal_std    : decimal(6, 4)     # temporal standard deviation
    cutoff          : decimal(6, 4)     # standard deviation cutoff
    """

    @rowproperty
    def visual(self):
        from fnn.model.stimuli import VisualNlm

        spatial_std, temporal_std, cutoff = self.fetch1("spatial_std", "temporal_std", "cutoff")

        return VisualNlm(
            bound=(Bound & self).link.nn,
            spatial_std=float(spatial_std),
            temporal_std=float(temporal_std),
            cutoff=float(cutoff),
        )


# -- Stimulus --


@schema.link
class Stimulus:
    links = [VisualNlm]
    name = "stimulus"
    comment = "stimulus module"


# ----------------------------- Descent -----------------------------

# -- Descent Interface --


class DescentType:
    """Descent"""

    @rowproperty
    def compute(self):
        """
        Returns
        -------
        foundation.fnn.compute_descent.DescentType (row)
            compute descent
        """
        raise NotImplementedError()


# -- Descent Types --


@schema.lookup
class VisualReconstruction(DescentType):
    definition = """
    -> stimulus.Video
    -> recording.TrialFilterSet
    sample_trial        : bool              # sample trial during optimization
    sample_stream       : bool              # sample stream during optimization
    burnin_frames       : int unsigned      # initial losses discarded
    stimulus_penalty    : decimal(9, 6)     # stimulus penalty weight
    """

    @rowproperty
    def compute(self):
        from foundation.fnn.compute_descent import VisualReconstruction

        return VisualReconstruction & self


# -- Descent --


@schema.link
class Descent:
    links = [VisualReconstruction]
    name = "descent"
    comment = "gradient descent"


# -- Computed Descent --


@schema.lookup
class DescentSteps:
    definition = """
    steps           : int unsigned      # descent steps per epoch
    """


@schema.computed
class VisualNetworkDescent:
    definition = """
    -> NetworkModel
    -> Descent
    -> Stimulus
    -> Optimizer
    -> Scheduler
    -> DescentSteps
    -> utility.Resolution
    ---
    frames          : int unsigned      # video frames
    channels        : int unsigned      # video channels
    video           : longblob          # [frames, height, width, channels] -- dtype=np.uint8
    """

    def make(self, key):
        from foundation.fnn.compute_descent import VisualNetworkDescent

        # compute video
        key["video"] = (VisualNetworkDescent & key).video

        # video size
        key["frames"], _, _, key["channel"] = key["video"].shape

        # insert
        self.insert1(key)
