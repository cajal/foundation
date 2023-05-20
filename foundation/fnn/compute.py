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

    @rowproperty
    def module(self):
        import torch
        from foundation.fnn.network import Network

        devices = torch.cuda.device_count()
        devices = list(range(devices))
        with torch.random.fork_rng(devices):

            torch.manual_seed(self.key.fetch("seed"))
            return (Network & self.key).link.module


@keys
class TrainVisualNetwork:
    """Train Visual Network"""

    @property
    def key_list(self):
        return [
            fnn.Network * fnn.Model.VisualModel & fnn.NetworkSet.Member,
        ]

    @rowmethod
    def train(self):
        import torch.distributed as dist
        from fnn.train.parallel import ParameterGroup
        from foundation.fnn.network import Network, NetworkSet
        from foundation.fnn.train import State, Scheduler, Optimizer, Loader, Objective
        from foundation.fnn.cache import NetworkModelInfo as Info, NetworkModelCheckpoint as Checkpoint

        if dist.is_initialized():
            size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            assert instances == 1
            size = 1
            rank = 0

        key = merge(self.key, fnn.Model.VisualModel)

        states = (State & key).link.network_keys
        module = (states & key).module.to(device="cuda")

        nid, mid, seed, cycle, instances = key.fetch1("network_id", "model_id", "seed", "cycle", "instances")
        checkpoints = Checkpoint & {"model_id": mid} & "rank >= 0" & f"rank < {size}"

        if checkpoints:
            assert len(checkpoints) == size
            assert len(U("epoch") & checkpoints) == 1

            logger.info("Restarting from checkpoint")
            c = (checkpoints & key & {"rank": rank}).load()

            optimizer = c["optimizer"]
            module.load_state_dict(c["state_dict"])

        else:
            if cycle > 0:
                logger.info("Continuing training")
                # TODO
                raise NotImplementedError()
            else:
                logger.info("Initializing training")

            scheduler = (Scheduler & key).link.scheduler
            scheduler._init(epoch=0, cycle=cycle)

            optimizer = (Optimizer & key).link.optimizer
            optimizer._init(scheduler=scheduler)

        dataset = (Network & key).link.dataset

        loader = (Loader & key).link.loader
        loader._init(dataset=dataset)

        objective = (Objective & key).link.objective
        objective._init(module=module)

        _groups = []
        groups = []

        if size > 1:
            cgroup = np.arange(size)
            cgroup = dist.new_group(cgroup.tolist())
            _groups.append([["core"], cgroup])

        if instances > 1:
            igroup = np.arange(instances) + rank // instances * instances
            igroup = dist.new_group(igroup.tolist())
            _groups.append([["perspective", "modulation", "readout", "reduce", "unit"], igroup])

        for mods, group in _groups:
            p = dict()
            for mod in mods:
                m = getattr(module, mod)
                if not m.frozen:
                    p = dict(**p, **{f"{mod}.{k}": v for k, v in m.named_parameters()})
            if p:
                groups.append(ParameterGroup(parameters=p, group=group))

        for epoch, info in optimizer.optimize(
            loader=loader,
            objective=objective,
            parameters=module.named_parameters(),
            groups=groups,
            seed=seed,
        ):
            Info.fill(
                network_id=nid,
                model_id=mid,
                rank=rank,
                epoch=epoch,
                info=info,
            )
            Checkpoint.fill(
                network_id=nid,
                model_id=mid,
                rank=rank,
                epoch=epoch,
                optimizer=optimizer,
                state_dict=module.state_dict(),
            )


@keys
class TrainVisualModel:
    """Train Visual Model"""

    @property
    def key_list(self):
        return [
            fnn.VisualModel,
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
            key = TrainVisualNetwork & {"network_id": network_id, "model_id": model_id}
            key.train()

    @staticmethod
    def _fn(rank, size, keys):
        kwargs = keys[rank]
        TrainVisualModel._train(rank=rank, size=size, **kwargs)

    @rowmethod
    def train(self):
        from torch.cuda import device_count
        from torch import multiprocessing as mp
        from foundation.fnn.train import Scheduler
        from foundation.fnn.network import NetworkSet
        from foundation.fnn.cache import NetworkModelCheckpoint as Checkpoint

        nets = merge((NetworkSet & self.key).members * self.key, fnn.Model.VisualModel)
        nets = nets.fetch("network_id", "model_id", as_dict=True, order_by="network_id")

        keys = nets * self.key.fetch1("instances")
        size = len(keys)
        assert device_count() >= size

        mp.spawn(
            TrainVisualModel._fn,
            args=(size, keys),
            nprocs=size,
            join=True,
        )

        key = merge(self.key, fnn.Model.VisualModel)
        epochs = (Scheduler & key).link.epochs
        checkpoints = Checkpoint & nets & "rank >= 0" & f"rank < {size}" & {"epoch": epochs - 1}
        assert len(checkpoints) == len(nets)

        keys = U("network_id", "model_id").aggr(checkpoints, rank="min(rank)").fetch(as_dict=True)
        for key in keys:
            yield key["network_id"], (Checkpoint & key).load()["state_dict"]
