import numpy as np
from djutils import keys, merge, rowmethod, rowproperty, MissingError
from foundation.utils import logger
from foundation.virtual import utility, stimulus, fnn


# ----------------------------- Descent -----------------------------

# -- Descent Interface --


class DescentType:
    """Descent"""

    @rowmethod
    def stimulus_objective(self, network_id, unit_index=None):
        """
        Parameters
        ----------
        network_id : str
            key (foundation.network.Network)
        unit_index : int | slice | 1D array | None
            unit index

        Returns
        -------
        fnn.train.objectives.StimulusObjective
            stimulus objective
        int
            stimulus channels
        int
            stimulus frames
        """
        raise NotImplementedError()


# -- Descent Types --


@keys
class VisualReconstruction(DescentType):
    """Visual Reconstruction"""

    @property
    def keys(self):
        return [
            fnn.VisualReconstruction,
        ]

    @rowmethod
    def stimulus_objective(self, network_id, unit_index=None):
        from foundation.fnn.data import Data
        from foundation.fnn.network import Network
        from foundation.utils.response import truncate
        from fnn.train.objectives import Reconstruction

        # network data
        data_id = (Network & {"network_id": network_id}).link.data_id

        # compute data
        data = (Data & {"data_id": data_id}).link.compute

        # stimulus channels
        channels = data.stimuli

        # visual trials
        trial_ids = data.visual_trial_ids(
            video_id=self.item["video_id"],
            trial_filterset_id=self.item["trial_filterset_id"],
        )

        # raise error if no trials
        if not trial_ids:
            raise MissingError("No trials found")

        # load traces
        def load(fn):
            traces = fn(trial_ids=trial_ids, use_cache=True)
            traces = truncate(*traces, tolerance=1)
            return np.stack(traces, axis=1)

        # stimulus objective
        objective = Reconstruction(
            trial_perspectives=load(data.trial_perspectives),
            trial_modulations=load(data.trial_modulations),
            trial_units=load(data.trial_units),
            unit_index=unit_index,
            sample_trial=self.item["sample_trial"],
            sample_stream=self.item["sample_stream"],
            burnin_frames=self.item["burnin_frames"],
            stimulus_penalty=self.item["stimulus_penalty"],
        )

        # stimulus frames
        frames = objective.frames

        return objective, channels, frames


# ----------------------------- Visual Descent -----------------------------

# -- Visual Descent Interfase --


class VisualDescentType:
    """Visual Descent"""

    @rowproperty
    def video(self):
        """
        Returns
        -------
        4D array
            [frames, height, width, channels], dtype=np.uint8
        """


@keys
class VisualNetworkDescent(DescentType):
    """Visual Reconstruction"""

    @property
    def keys(self):
        return [
            fnn.NetworkModel,
            fnn.Descent,
            fnn.Stimulus,
            fnn.Optimizer,
            fnn.Scheduler,
            fnn.DescentSteps,
            utility.Resolution,
        ]

    @staticmethod
    def _fn(_, item):
        from fnn.train.loaders import EmptyLoader
        from foundation.fnn.descent import Descent, Stimulus
        from foundation.fnn.train import Optimizer, Scheduler
        from foundation.fnn.model import NetworkModel
        from foundation.fnn import progress

        # stimulus objective
        descent = (Descent & item).link.compute
        objective, channels, frames = descent.stimulus_objective(network_id=item["network_id"])

        # stimulus module
        stimulus = (Stimulus & item).link.visual
        stimulus._init(channels=channels, frames=frames, height=item["height"], width=item["width"])

        # network module
        network = (NetworkModel & item).model

        # initialize objective
        objective._init(
            network=network.to(device="cuda"),
            stimulus=stimulus.to(device="cuda"),
        )

        # checkpoint
        checkpoint = progress.VisualNetworkDescentCheckpoint & item

        if checkpoint:
            logger.info("Reloading from checkpoint")

            # load checkpoint
            prev = checkpoint.load(device="cuda")

            # reload stimulus
            stimulus.load_state_dict(prev["state_dict"])

            # optimizer
            optimizer = prev["optimizer"]

        else:
            logger.info("Starting descent")

            # scheduler
            scheduler = (Scheduler & item).link.scheduler
            scheduler._init(epoch=0, cycle=0)

            # optimizer
            optimizer = (Optimizer & item).link.optimizer
            optimizer._init(scheduler=scheduler)

        # parameters
        params = stimulus.named_parameters()

        # loader
        loader = EmptyLoader(training_size=item["steps"], validation_size=0)

        # descent epochs
        for epoch, info in optimizer.optimize(loader=loader, objective=objective, parameters=params):

            # video
            video = stimulus.video
            assert np.array_equal(video.shape, [frames, item["height"], item["width"], channels])

            # save epoch info
            progress.VisualNetworkDescentInfo.fill({"epoch": epoch, "info": info, **item})

            # save checkpoint
            checkpoint = {"optimizer": optimizer, "state_dict": stimulus.state_dict(), "video": video}
            progress.VisualNetworkDescentCheckpoint.fill({"epoch": epoch, "checkpoint": checkpoint, **item})

        # register done
        progress.VisualNetworkDescentDone.insert1({"epoch": epoch, **item})

    @rowproperty
    def video(self):
        from torch.multiprocessing import spawn
        from foundation.fnn.progress import VisualNetworkDescentDone, VisualNetworkDescentCheckpoint

        # row item
        item = self.item

        # compute in separate process to enable checkpoint saving
        conn = self.key.connection
        conn.close()
        spawn(
            VisualNetworkDescent._fn,
            args=(item,),
            nprocs=1,
            join=True,
        )
        conn.connect()

        # return computed video
        key = (progress.VisualNetworkDescentDone & self.item).fetch1()
        return (progress.VisualNetworkDescentCheckpoint & key).load()["video"]
