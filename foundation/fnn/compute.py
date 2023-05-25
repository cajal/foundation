import numpy as np
import pandas as pd
from tqdm import tqdm
from datajoint import U
from djutils import keys, merge, keyproperty, rowproperty, rowmethod, cache_rowproperty
from foundation.utils import logger
from foundation.virtual import stimulus, recording, fnn


# ----------------------------- Data -----------------------------


@keys
class VisualScan:
    """Visual Scan"""

    @property
    def key_list(self):
        return [
            fnn.VisualScan,
        ]

    @rowproperty
    def stimuli_key(self):
        return merge(
            self.key.proj(spec_id="stimuli_id"),
            fnn.Spec.VideoSpec,
        ).fetch1()

    @rowproperty
    def perspectives_key(self):
        return merge(
            self.key.proj(spec_id="perspectives_id"),
            fnn.Spec.TraceSpec,
            recording.ScanPerspectives,
            recording.ScanTrials,
        ).fetch1()

    @rowproperty
    def modulations_key(self):
        return merge(
            self.key.proj(spec_id="modulations_id"),
            fnn.Spec.TraceSpec,
            recording.ScanModulations,
            recording.ScanTrials,
        ).fetch1()

    @rowproperty
    def units_key(self):
        return merge(
            self.key.proj(spec_id="units_id"),
            fnn.Spec.TraceSpec,
            recording.ScanUnits,
            recording.ScanTrials,
        ).fetch1()

    @rowproperty
    def perspectives_transform(self):
        from foundation.recording.compute import StandardizeTraces

        return (StandardizeTraces & self.perspectives_key).transform

    @rowproperty
    def modulations_transform(self):
        from foundation.recording.compute import StandardizeTraces

        return (StandardizeTraces & self.modulations_key).transform

    @rowproperty
    def units_transform(self):
        from foundation.recording.compute import StandardizeTraces

        return (StandardizeTraces & self.units_key).transform

    @rowproperty
    def trials(self):
        from foundation.recording.trial import Trial, TrialSet

        key = merge(self.key, recording.ScanTrials).fetch1()

        return Trial & (TrialSet & key).members

    @rowproperty
    def all_trials(self):
        from foundation.recording.trial import Trial, TrialSet

        key = [self.perspectives_key, self.modulations_key, self.units_key]
        key = merge(recording.TraceSet.Member & key, recording.TraceTrials)
        key = (U("trialset_id") & key).fetch1()

        return Trial & (TrialSet & key).members

    @rowproperty
    def samples(self):
        from foundation.recording.trial import TrialSamples

        trials = merge(
            self.trials,
            TrialSamples & self.key,
        )
        trial_id, samples = trials.fetch("trial_id", "samples", order_by="trial_id")
        index = pd.Index(trial_id, name="trial_id")

        return pd.Series(data=samples, index=index)

    @rowproperty
    def stimuli(self):
        from fnn.data import NpyFile

        key = self.stimuli_key
        trials = merge(
            self.trials,
            recording.TrialVideo,
            stimulus.ResizedVideo & key,
            recording.ResampledVideo & key,
        )
        trial_id, video, imap = trials.fetch("trial_id", "video", "index", order_by="trial_id")
        index = pd.Index(trial_id, name="trial_id")
        data = [NpyFile(v, indexmap=np.load(i), dtype=np.uint8) for v, i in zip(video, tqdm(imap, desc="Video"))]

        return pd.Series(data=data, index=index)

    @rowmethod
    def _traces(self, key):
        from foundation.recording.compute import StandardizeTraces
        from fnn.data import NpyFile

        transform = (StandardizeTraces & key).transform
        trials = merge(
            self.trials,
            recording.ResampledTraces & key,
        )
        trial_id, traces = trials.fetch("trial_id", "traces", order_by="trial_id")
        index = index = pd.Index(trial_id, name="trial_id")
        data = [NpyFile(t, transform=transform, dtype=np.float32) for t in tqdm(traces, desc="Traces")]

        return pd.Series(data=data, index=index)

    @rowproperty
    def trainset(self):
        from fnn.data import Dataset

        data = [
            self.samples.rename("samples"),
            self.stimuli.rename("stimuli"),
            self._traces(self.perspectives_key).rename("perspectives"),
            self._traces(self.modulations_key).rename("modulations"),
            self._traces(self.units_key).rename("units"),
        ]
        df = pd.concat(data, axis=1, join="outer")
        assert not df.isnull().values.any()

        return Dataset(df)

    @rowproperty
    def network_sizes(self):

        stimuli = merge(self.trials, recording.TrialVideo, stimulus.VideoInfo)
        perspectives = recording.TraceSet & self.perspectives_key
        modulations = recording.TraceSet & self.modulations_key
        units = recording.TraceSet & self.units_key

        return {
            "stimuli": (U("channels") & stimuli).fetch1("channels"),
            "perspectives": perspectives.fetch1("members"),
            "modulations": modulations.fetch1("members"),
            "units": units.fetch1("members"),
        }


@keys
class VisualScanInputs:
    """Visual Scan Inputs"""

    @property
    def key_list(self):
        return [
            fnn.VisualScan,
            stimulus.Video,
        ]

    @rowproperty
    def trials(self):
        trials = (VisualScan & self.key).all_trials.proj()
        return merge(trials, recording.TrialBounds, recording.TrialVideo) & self.key

    @rowmethod
    def stimuli(self):
        """
        Yields
        ------
        4D array -- [batch_size, height, width, channels]
            video frame
        """
        from foundation.utils.resample import truncate, flip_index
        from foundation.utility.resample import Rate
        from foundation.stimulus.compute import ResizeVideo
        from foundation.recording.compute import ResampleTrial

        # load video
        key = (VisualScan & self.key).stimuli_key
        video = (ResizeVideo & self.key & key).video

        # load trials
        trials = self.trials
        if trials:

            # video indexes based on trial timing
            with cache_rowproperty():
                trials = trials.fetch("KEY", order_by="start")
                indexes = [(ResampleTrial & trial & self.key).video_index for trial in trials]

            indexes = truncate(*indexes)
            indexes = np.stack(indexes, axis=0)

            if not np.diff(indexes, axis=0).any():
                indexes = indexes[:1]

        else:
            # video indexes based on expected timing
            times = video.times
            time_scale = merge(self.key, recording.ScanVideoTiming).fetch1("time_scale")
            period = (Rate & self.key).link.period

            indexes = flip_index(times * time_scale, period)[None]

        # yield video frames
        varray = video.array
        for i in indexes.T:
            yield varray[i]

    @rowmethod
    def perspectives(self):
        """
        Yields
        ------
        2D array -- [batch_size, traces]
            perspective frame
        """
        from foundation.utils.resample import truncate
        from foundation.recording.compute import ResampleTraces

        # load trials
        trials = self.trials
        if not trials:
            return

        with cache_rowproperty():
            # traceset keys
            trials = trials.fetch("KEY", order_by="start")
            key = (VisualScan & self.key).perspectives_key

            # traceset transformation
            transform = (VisualScan & self.key).perspectives_transform

            # load and transform traceset
            inputs = ((ResampleTraces & trial & key).trial for trial in trials)
            inputs = (transform(i) for i in truncate(*inputs))
            inputs = np.stack(list(inputs), axis=1)

        # yield traceset frames
        def generate():
            yield from inputs

        return generate()

    @rowmethod
    def modulations(self):
        """
        Yields
        ------
        2D array -- [batch_size, traces]
            modulation frame
        """
        from foundation.utils.resample import truncate
        from foundation.recording.compute import ResampleTraces

        # load trials
        trials = self.trials
        if not trials:
            return

        with cache_rowproperty():
            # traceset keys
            trials = trials.fetch("KEY", order_by="start")
            key = (VisualScan & self.key).modulations_key

            # traceset transformation
            transform = (VisualScan & self.key).modulations_transform

            # load and transform traceset
            inputs = ((ResampleTraces & trial & key).trial for trial in trials)
            inputs = (transform(i) for i in truncate(*inputs))
            inputs = np.stack(list(inputs), axis=1)

        # yield traceset frames
        def generate():
            yield from inputs

        return generate()


# ----------------------------- State -----------------------------


@keys
class RandomNetworkState:
    """Random Network State"""

    @property
    def key_list(self):
        return [
            fnn.Network,
            fnn.RandomState,
        ]

    @rowmethod
    def build(self, initialize=True):
        import torch
        from foundation.fnn.network import Network

        devices = list(range(torch.cuda.device_count()))
        with torch.random.fork_rng(devices):

            if initialize:
                seed = self.key.fetch1("seed")
                torch.manual_seed(seed)
                logger.info(f"Initializing network with random seed {seed}")

            return (Network & self.key).link.module


# ----------------------------- Training -----------------------------


@keys
class _TrainNetwork:
    """Train Network"""

    @property
    def key_list(self):
        return [
            fnn.Network,
            fnn.State,
            fnn.Loader,
            fnn.Objective,
            fnn.Optimizer,
            fnn.Scheduler,
        ]

    @rowmethod
    def train(self, model_id, cycle=0, seed=0, instances=1, rank=0, size=1):
        from torch import device
        from torch.cuda import current_device
        from foundation.fnn.network import Network
        from foundation.fnn.model import Model
        from foundation.fnn.train import State, Scheduler, Optimizer, Loader, Objective
        from foundation.fnn.cache import ModelNetworkInfo, ModelNetworkCheckpoint

        network_id = self.key.fetch1("network_id")
        checkpoint = ModelNetworkCheckpoint & {"model_id": model_id} & "rank >= 0" & f"rank < {size}"
        init = not checkpoint and cycle == 0
        cuda = device("cuda", current_device())

        nets = (State & self.key).link.network_keys
        module = (nets & self.key).build(initialize=init).to(device=cuda)

        if checkpoint:
            assert len(checkpoint) == size
            assert len(U("epoch") & checkpoint) == 1

            logger.info("Reloading from previous checkpoint")
            prev = (checkpoint & self.key & {"rank": rank}).load(device=cuda)
            module.load_state_dict(prev["state_dict"])
            optimizer = prev["optimizer"]

        else:
            if cycle == 0:
                logger.info("Initializing training")
            else:
                logger.info("Reloading from previous cycle")
                prev = (Model & {"model_id": model_id}).link.previous_networks
                prev = (prev & self.key).parameters(device=cuda)
                module.load_state_dict(prev)

            scheduler = (Scheduler & self.key).link.scheduler
            scheduler._init(epoch=0, cycle=cycle)

            optimizer = (Optimizer & self.key).link.optimizer
            optimizer._init(scheduler=scheduler)

        dataset = (Network & self.key).link.trainset
        loader = (Loader & self.key).link.loader
        loader._init(dataset=dataset)

        objective = (Objective & self.key).link.objective
        objective._init(module=module)

        params = module.named_parameters()
        groups = module.parallel_groups(instances=instances)

        for epoch, info in optimizer.optimize(
            loader=loader, objective=objective, parameters=params, groups=groups, seed=seed
        ):
            ModelNetworkInfo.fill(
                model_id=model_id,
                network_id=network_id,
                rank=rank,
                epoch=epoch,
                info=info,
            )
            ModelNetworkCheckpoint.fill(
                model_id=model_id,
                network_id=network_id,
                rank=rank,
                epoch=epoch,
                optimizer=optimizer,
                state_dict=module.state_dict(),
            )


@keys
class TrainNetworkSet:
    """Train Network Set"""

    @property
    def key_list(self):
        nets = fnn.NetworkSet.Member * fnn.Network.VisualNetwork
        nets = U("networkset_id").aggr(nets, c="count(distinct core_id)", s="count(distinct streams)")
        return [
            fnn.NetworkSetModel & (nets & "c=1 and s=1").fetch("KEY"),
        ]

    @staticmethod
    def _train(rank, size, model_id, network_id, port=23456, backend="nccl"):
        from torch.cuda import device
        from torch.distributed import init_process_group

        with device(rank):
            init_process_group(
                backend=backend,
                init_method=f"tcp://0.0.0.0:{port}",
                rank=rank,
                world_size=size,
            )
            key = fnn.Network * fnn.Model.NetworkSetModel & {"model_id": model_id, "network_id": network_id}
            cycle, seed, instances = key.fetch1("cycle", "seed", "instances")
            trainer = _TrainNetwork & key
            trainer.train(
                model_id=model_id,
                cycle=cycle,
                seed=seed,
                instances=instances,
                rank=rank,
                size=size,
            )

    @staticmethod
    def _fn(rank, size, port, keys):
        kwargs = keys[rank]
        TrainNetworkSet._train(rank=rank, size=size, port=port, **kwargs)

    @rowmethod
    def train(self):
        from random import randint
        from torch.cuda import device_count
        from torch.multiprocessing import spawn
        from foundation.fnn.train import Scheduler
        from foundation.fnn.network import NetworkSet
        from foundation.fnn.cache import ModelNetworkCheckpoint as Checkpoint

        nets = merge((NetworkSet & self.key).members * self.key, fnn.Model.NetworkSetModel)
        nets = nets.fetch("network_id", "model_id", as_dict=True, order_by="network_id")

        keys = nets * self.key.fetch1("instances")
        size = len(keys)
        assert device_count() >= size

        conn = self.key.connection
        conn.close()
        port = randint(10000, 60000)
        spawn(
            TrainNetworkSet._fn,
            args=(size, port, keys),
            nprocs=size,
            join=True,
        )
        conn.connect()

        key = merge(self.key, fnn.Model.NetworkSetModel)
        epochs = (Scheduler & key).link.epochs
        checkpoints = Checkpoint & nets & "rank >= 0" & f"rank < {size}" & {"epoch": epochs - 1}
        assert len(checkpoints) == size

        keys = U("network_id", "model_id").aggr(checkpoints, rank="min(rank)").fetch(as_dict=True)
        for key in keys:
            yield key["network_id"], (Checkpoint & key).load()["state_dict"]


@keys
class TrainNetwork:
    """Train Network"""

    @property
    def key_list(self):
        return [
            fnn.NetworkModel,
        ]

    @staticmethod
    def _train(rank, size, model_id, port=23456, backend="nccl"):
        from torch.cuda import device
        from torch.distributed import init_process_group

        with device(rank):
            init_process_group(
                backend=backend,
                init_method=f"tcp://0.0.0.0:{port}",
                rank=rank,
                world_size=size,
            )
            key = fnn.Model.NetworkModel & {"model_id": model_id}
            cycle, seed, instances = key.fetch1("cycle", "seed", "instances")
            trainer = _TrainNetwork & key
            trainer.train(
                model_id=model_id,
                cycle=cycle,
                seed=seed,
                instances=instances,
                rank=rank,
                size=size,
            )

    @rowmethod
    def train(self):
        from random import randint
        from torch.cuda import device_count
        from torch.multiprocessing import spawn
        from foundation.fnn.train import Scheduler
        from foundation.fnn.cache import ModelNetworkCheckpoint as Checkpoint

        key = merge(self.key, fnn.Model.NetworkModel)
        size, model_id = key.fetch1("instances", "model_id")
        assert device_count() >= size

        conn = self.key.connection
        conn.close()
        port = randint(10000, 60000)
        spawn(
            TrainNetwork._train,
            args=(size, model_id, port),
            nprocs=size,
            join=True,
        )
        conn.connect()

        epochs = (Scheduler & key).link.epochs
        checkpoints = Checkpoint & key & "rank >= 0" & f"rank < {size}" & {"epoch": epochs - 1}
        assert len(checkpoints) == size

        key = U("network_id", "model_id").aggr(checkpoints, rank="min(rank)").fetch1()
        return key["network_id"], (Checkpoint & key).load()["state_dict"]
