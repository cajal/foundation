import numpy as np
from djutils import keys, merge, rowmethod, rowproperty, MissingError
from foundation.utils import logger
from foundation.virtual import utility, stimulus, fnn


# ----------------------------- Descent -----------------------------

# -- Descent Interface --


class DescentType:
    """Descent"""

    @rowmethod
    def stimulus_objective(self, network_id):
        """
        Parameters
        ----------
        network_id : str
            key (foundation.network.Network)

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
class Excitation(DescentType):
    """Visual Excitation"""

    @property
    def keys(self):
        return [
            fnn.Excitation,
        ]

    @rowmethod
    def stimulus_objective(self, network_id):
        from foundation.fnn.data import Data
        from foundation.fnn.network import Network
        from fnn.train.objectives import Excitation

        # stimulus channels
        data_id = (Network & {"network_id": network_id}).link.data_id
        channels = (Data & {"data_id": data_id}).link.compute.stimuli

        # stimulus frames
        frames = self.item["stimulus_frames"]

        # stimulus objective
        objective = Excitation(
            temperature=self.item["temperature"],
            sample_stream=self.item["sample_stream"],
            burnin_frames=self.item["burnin_frames"],
            stimulus_penalty=self.item["stimulus_penalty"],
        )

        return objective, channels, frames


@keys
class VisualReconstruction(DescentType):
    """Visual Reconstruction"""

    @property
    def keys(self):
        return [
            fnn.VisualReconstruction,
        ]

    @rowmethod
    def stimulus_objective(self, network_id):
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
        raise NotImplementedError()

    @property
    def progress_tables(self):
        """
        Returns
        -------
        checkpoint_table : foundation.fnn.progress.Checkpoint
            checkpoint table
        info_table : foundation.fnn.progress.Info
            info table
        done_table : datajoint.UserTable
            done table
        """
        raise NotImplementedError()

    @staticmethod
    def _fn(
        _,
        network_id,
        model_id,
        descent_id,
        stimulus_id,
        optimizer_id,
        scheduler_id,
        steps,
        height,
        width,
        progress_key,
        checkpoint_table,
        info_table,
        done_table,
        unit_index=None,
    ):
        """
        Parameters
        ----------
        network_id : str
            key (foundation.fnn.network.Network)
        model_id : str
            key (foundation.fnn.model.Model)
        descent_id : str
            key (foundation.fnn.descent.Descent)
        stimulus_id : str
            key (foundation.fnn.descent.Stimulus)
        optimizer_id : str
            key (foundation.fnn.train.Optimizer)
        scheduler_id : str
            key (foundation.fnn.train.Scheduler)
        steps : int
            descent steps per epoch
        height : int
            video height
        width : int
            video width
        progress_key : dict
            key that restricts progress tables
        checkpoint_table : str
            name of the checkpoint table in foundation.fnn.progress
        info_table : str
            name of the info table in foundation.fnn.progress
        done_table : str
            name of the done table in foundation.fnn.progress
        unit_index : int | 1D array | None
            unit index
        """
        from fnn.train.loaders import EmptyLoader
        from foundation.fnn.descent import Descent, Stimulus
        from foundation.fnn.train import Optimizer, Scheduler
        from foundation.fnn.model import NetworkModel
        from foundation.fnn import progress

        # stimulus objective
        descent = (Descent & {"descent_id": descent_id}).link.compute
        objective, channels, frames = descent.stimulus_objective(network_id=network_id)

        # stimulus module
        stimulus = (Stimulus & {"stimulus_id": stimulus_id}).link.visual
        stimulus._init(channels=channels, frames=frames, height=height, width=width)

        # network module
        network = (NetworkModel & {"network_id": network_id, "model_id": model_id}).model

        # initialize objective
        objective._init(
            network=network.to(device="cuda"),
            stimulus=stimulus.to(device="cuda"),
            unit_index=unit_index,
        )

        # progress tables
        checkpoint_table = getattr(progress, checkpoint_table)
        info_table = getattr(progress, info_table)
        done_table = getattr(progress, done_table)

        # checkpoint
        checkpoint = checkpoint_table & progress_key

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
            scheduler = (Scheduler & {"scheduler_id": scheduler_id}).link.scheduler
            scheduler._init(epoch=0, cycle=0)

            # optimizer
            optimizer = (Optimizer & {"optimizer_id": optimizer_id}).link.optimizer
            optimizer._init(scheduler=scheduler)

        # parameters
        params = stimulus.named_parameters()

        # loader
        loader = EmptyLoader(training_size=steps, validation_size=0)

        # descent epochs
        for epoch, info in optimizer.optimize(loader=loader, objective=objective, parameters=params):

            # video
            video = stimulus.video
            assert np.array_equal(video.shape, [frames, height, width, channels])

            # save epoch info
            info_table.fill({"epoch": epoch, "info": info, **progress_key})

            # save checkpoint
            checkpoint = {"optimizer": optimizer, "state_dict": stimulus.state_dict(), "video": video}
            checkpoint_table.fill({"epoch": epoch, "checkpoint": checkpoint, **progress_key})

        # register done
        done_table.insert1({"epoch": epoch, **progress_key})


@keys
class VisualNetworkDescent(VisualDescentType):
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

    @rowproperty
    def video(self):
        from torch.multiprocessing import spawn
        from foundation.fnn import progress

        # row item
        item = self.item

        # compute in separate process to enable checkpoint saving
        conn = self.key.connection
        conn.close()
        spawn(
            self._fn,
            args=(
                item["network_id"],
                item["model_id"],
                item["descent_id"],
                item["stimulus_id"],
                item["optimizer_id"],
                item["scheduler_id"],
                item["steps"],
                item["height"],
                item["width"],
                item,
                "VisualNetworkDescentCheckpoint",
                "VisualNetworkDescentInfo",
                "VisualNetworkDescentDone",
            ),
            nprocs=1,
            join=True,
        )
        conn.connect()

        # return computed video
        key = (progress.VisualNetworkDescentDone & self.item).fetch1()
        return (progress.VisualNetworkDescentCheckpoint & key).load()["video"]


@keys
class VisualUnitDescent(VisualDescentType):
    """Visual Reconstruction"""

    @property
    def keys(self):
        return [
            fnn.NetworkModel,
            fnn.NetworkUnit,
            fnn.Descent,
            fnn.Stimulus,
            fnn.Optimizer,
            fnn.Scheduler,
            fnn.DescentSteps,
            utility.Resolution,
        ]

    @rowproperty
    def video(self):
        from torch.multiprocessing import spawn
        from foundation.fnn import progress

        # row item
        item = self.item

        # compute in separate process to enable checkpoint saving
        conn = self.key.connection
        conn.close()
        spawn(
            self._fn,
            args=(
                item["network_id"],
                item["model_id"],
                item["descent_id"],
                item["stimulus_id"],
                item["optimizer_id"],
                item["scheduler_id"],
                item["steps"],
                item["height"],
                item["width"],
                item,
                "VisualUnitDescentCheckpoint",
                "VisualUnitDescentInfo",
                "VisualUnitDescentDone",
                item["unit_index"],
            ),
            nprocs=1,
            join=True,
        )
        conn.connect()

        # return computed video
        key = (progress.VisualUnitDescentDone & self.item).fetch1()
        return (progress.VisualUnitDescentCheckpoint & key).load()["video"]
