from djutils import rowproperty
from foundation.virtual import utility, stimulus, recording
from foundation.fnn.shared import Bound
from foundation.fnn.train import Optimizer, Scheduler
from foundation.fnn.network import NetworkUnit
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
    init_value      : decimal(6, 4)     # initial pixel value
    spatial_std     : decimal(6, 4)     # spatial standard deviation
    temporal_std    : decimal(6, 4)     # temporal standard deviation
    cutoff          : decimal(6, 4)     # standard deviation cutoff
    """

    @rowproperty
    def visual(self):
        from fnn.model.stimuli import VisualNlm

        init, spatial, temporal, cutoff = self.fetch1("init_value", "spatial_std", "temporal_std", "cutoff")

        return VisualNlm(
            bound=(Bound & self).link.nn,
            init_value=float(init),
            spatial_std=float(spatial),
            temporal_std=float(temporal),
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
class Excitation(DescentType):
    definition = """
    sample_stream       : bool              # sample stream during optimization
    burnin_frames       : int unsigned      # initial losses discarded
    stimulus_frames     : int unsigned      # stimulus frames
    stimulus_penalty    : decimal(9, 6)     # stimulus penalty weight
    temperature         : decimal(9, 6)     # exponential temperature
    """

    @rowproperty
    def compute(self):
        from foundation.fnn.compute_descent import Excitation

        return Excitation & self


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
    links = [Excitation, VisualReconstruction]
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
        video = (VisualNetworkDescent & key).video
        frames, _, _, channels = video.shape

        # insert
        self.insert1(dict(key, frames=frames, channels=channels, video=video))


@schema.computed
class VisualUnitDescent:
    definition = """
    -> NetworkModel
    -> NetworkUnit
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
        from foundation.fnn.compute_descent import VisualUnitDescent

        # compute video
        video = (VisualUnitDescent & key).video
        frames, _, _, channels = video.shape

        # insert
        self.insert1(dict(key, frames=frames, channels=channels, video=video))
