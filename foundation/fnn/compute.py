import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import chain
from datajoint import U
from djutils import keys, merge, rowproperty, rowmethod
from foundation.utils import logger
from foundation.virtual import fnn


@keys
class ResampledVisualRecording:
    """Load Preprocessed Data"""

    @property
    def key_list(self):
        return [
            fnn.VisualRecording,
            fnn.VisualSpec.ResampleVisual,
        ]

    @rowproperty
    def trials(self):
        from foundation.recording.trial import Trial, TrialSet

        key = merge(self.key, fnn.VisualRecording)
        return Trial & (TrialSet & key).members

    @rowproperty
    def trial_samples(self):
        from foundation.recording.trial import TrialSamples

        key = merge(self.key, fnn.VisualSpec.ResampleVisual)

        trials = merge(self.trials, TrialSamples & key)
        trial_id, samples = trials.fetch("trial_id", "samples", order_by="trial_id")

        return pd.Series(data=samples, index=pd.Index(trial_id, name="trial_id"))

    @rowproperty
    def trial_video(self):
        from foundation.recording.trial import TrialVideo
        from foundation.recording.cache import ResampledVideo
        from foundation.stimulus.cache import ResizedVideo
        from fnn.data import NpyFile

        key = merge(self.key, fnn.VisualSpec.ResampleVisual)

        trials = merge(self.trials, TrialVideo, ResizedVideo & key, ResampledVideo & key)
        trial_id, video, index = trials.fetch("trial_id", "video", "index", order_by="trial_id")

        data = [NpyFile(v, indexmap=np.load(i), dtype=np.uint8) for v, i in zip(video, tqdm(index, desc="Video"))]
        return pd.Series(data=data, index=pd.Index(trial_id, name="trial_id"))

    @rowmethod
    def traceset_key(self, suffix="p"):
        if suffix not in ["p", "m", "u"]:
            raise ValueError("Suffix must be one of {'p', 'm', 'u'}")

        proj = {f"{k}_id": f"{k}_id_{suffix}" for k in ["traceset", "offset", "resample", "standardize"]}
        key = merge(self.key, fnn.VisualRecording, fnn.VisualSpec.ResampleVisual).proj(..., **proj)

        attrs = ["traceset_id", "trialset_id", "standardize_id", "rate_id", "offset_id", "resample_id"]
        key = U(*attrs) & key

        return key.fetch1("KEY")

    @rowmethod
    def trial_traces(self, suffix="p"):
        from foundation.recording.compute import StandardizeTraces
        from foundation.recording.cache import ResampledTraces
        from fnn.data import NpyFile

        key = self.traceset_key(suffix)

        transform = (StandardizeTraces & key).transform

        trials = merge(self.trials, ResampledTraces & key & "finite")
        trial_id, traces = trials.fetch("trial_id", "traces", order_by="trial_id")

        data = [NpyFile(t, transform=transform, dtype=np.float32) for t in tqdm(traces, desc="Traces")]
        return pd.Series(data=data, index=pd.Index(trial_id, name="trial_id"))

    @rowproperty
    def dataset(self):
        from fnn.data import Dataset

        data = [
            self.trial_samples.rename("samples"),
            self.trial_video.rename("stimuli"),
            self.trial_traces("p").rename("perspectives"),
            self.trial_traces("m").rename("modulations"),
            self.trial_traces("u").rename("units"),
        ]
        df = pd.concat(data, axis=1, join="outer")
        assert not df.isnull().values.any()

        return Dataset(df)

    @rowproperty
    def sizes(self):
        from foundation.stimulus.video import VideoInfo
        from foundation.recording.trial import TrialVideo
        from foundation.recording.trace import TraceSet

        return {
            "stimuli": (U("channels") & merge(self.trials, TrialVideo, VideoInfo)).fetch1("channels"),
            "perspectives": (TraceSet & self.traceset_key("p")).fetch1("members"),
            "modulations": (TraceSet & self.traceset_key("m")).fetch1("members"),
            "units": (TraceSet & self.traceset_key("u")).fetch1("members"),
        }


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


# @keys
# class _TrainNetworkSet:
#     """Train Network Set -- Process"""

#     @property
#     def key_list(self):
#         return [
#             fnn.Model.NetworkSetModel * fnn.Network & fnn.NetworkSet.Member,
#         ]

#     @rowmethod
#     def train(self):
#         from torch import device
#         from torch.cuda import current_device
#         from torch.distributed import is_initialized, get_world_size, get_rank
#         from fnn.train.parallel import ParameterGroup
#         from foundation.fnn.network import Network, NetworkSet
#         from foundation.fnn.train import State, Scheduler, Optimizer, Loader, Objective
#         from foundation.fnn.cache import ModelNetworkInfo, ModelNetworkCheckpoint

#         if is_initialized():
#             size = get_world_size()
#             rank = get_rank()
#         else:
#             assert instances == 1
#             size = 1
#             rank = 0

#         key = merge(self.key, fnn.Model.NetworkSetModel)
#         mid, nid, seed, cycle, instances = key.fetch1("model_id", "network_id", "seed", "cycle", "instances")
#         checkpoint = ModelNetworkCheckpoint & {"model_id": mid} & "rank >= 0" & f"rank < {size}"

#         init = not checkpoint and cycle == 0
#         cuda = device("cuda", current_device())
#         nets = (State & key).link.network_keys
#         module = (nets & key).build(initialize=init).to(device=cuda)

#         if checkpoint:
#             logger.info("Reloading from previous checkpoint")

#             assert len(checkpoint) == size
#             assert len(U("epoch") & checkpoint) == 1

#             prev = (checkpoint & key & {"rank": rank}).load(device=cuda)
#             module.load_state_dict(prev["state_dict"])
#             optimizer = prev["optimizer"]

#         else:
#             if cycle == 0:
#                 logger.info("Initializing training")
#             else:
#                 logger.info("Reloading from previous cycle")
#                 # TODO
#                 raise NotImplementedError()

#             scheduler = (Scheduler & key).link.scheduler
#             scheduler._init(epoch=0, cycle=cycle)

#             optimizer = (Optimizer & key).link.optimizer
#             optimizer._init(scheduler=scheduler)

#         dataset = (Network & key).link.dataset

#         loader = (Loader & key).link.loader
#         loader._init(dataset=dataset)

#         objective = (Objective & key).link.objective
#         objective._init(module=module)

#         params = module.named_parameters()
#         groups = module.parallel_groups(instances=instances)

#         for epoch, info in optimizer.optimize(
#             loader=loader, objective=objective, parameters=params, groups=groups, seed=seed
#         ):
#             ModelNetworkInfo.fill(
#                 model_id=mid,
#                 network_id=nid,
#                 rank=rank,
#                 epoch=epoch,
#                 info=info,
#             )
#             ModelNetworkCheckpoint.fill(
#                 model_id=mid,
#                 network_id=nid,
#                 rank=rank,
#                 epoch=epoch,
#                 optimizer=optimizer,
#                 state_dict=module.state_dict(),
#             )


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
        from foundation.fnn.train import State, Scheduler, Optimizer, Loader, Objective
        from foundation.fnn.cache import ModelNetworkInfo, ModelNetworkCheckpoint

        checkpoint = ModelNetworkCheckpoint & {"model_id": model_id} & "rank >= 0" & f"rank < {size}"
        init = not checkpoint and cycle == 0
        cuda = device("cuda", current_device())

        nets = (State & self.key).link.network_keys
        module = (nets & self.key).build(initialize=init).to(device=cuda)

        if checkpoint:
            logger.info("Reloading from previous checkpoint")

            assert len(checkpoint) == size
            assert len(U("epoch") & checkpoint) == 1

            prev = (checkpoint & self.key & {"rank": rank}).load(device=cuda)
            module.load_state_dict(prev["state_dict"])
            optimizer = prev["optimizer"]

        else:
            if cycle == 0:
                logger.info("Initializing training")
            else:
                logger.info("Reloading from previous cycle")
                # TODO
                raise NotImplementedError()

            scheduler = (Scheduler & self.key).link.scheduler
            scheduler._init(epoch=0, cycle=cycle)

            optimizer = (Optimizer & self.key).link.optimizer
            optimizer._init(scheduler=scheduler)

        dataset = (Network & self.key).link.dataset

        loader = (Loader & self.key).link.loader
        loader._init(dataset=dataset)

        objective = (Objective & self.key).link.objective
        objective._init(module=module)

        params = module.named_parameters()
        groups = module.parallel_groups(instances=instances)

        network_id = self.fetch1("network_id")

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
        return [
            fnn.NetworkSetModel,
        ]

    @staticmethod
    def _train(network_id, model_id, rank, size, backend="nccl", port=23456):
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
        from time import sleep
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

        port = randint(10000, 60000)

        spawn(
            TrainNetworkSet._fn,
            args=(size, port, keys),
            nprocs=size,
            join=True,
        )

        sleep(4)
        key = merge(self.key, fnn.Model.NetworkSetModel)
        epochs = (Scheduler & key).link.epochs
        checkpoints = Checkpoint & nets & "rank >= 0" & f"rank < {size}" & {"epoch": epochs - 1}
        assert len(checkpoints) == size

        keys = U("network_id", "model_id").aggr(checkpoints, rank="min(rank)").fetch(as_dict=True)
        for key in keys:
            yield key["network_id"], (Checkpoint & key).load()["state_dict"]
