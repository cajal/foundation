from djutils import keys, merge, cache_rowproperty, U
from foundation.virtual.bridge import pipe_fuse, pipe_shared
from foundation.virtual import utility, stimulus, scan, recording, fnn


class _VisualScanData:
    """Visual Scan Data -- Base"""

    @property
    def unit_set(self):
        raise NotImplementedError()

    @property
    def unit_order(self):
        raise NotImplementedError()

    @property
    def datatype(self):
        raise NotImplementedError()

    def fill(self, training_tier=0, validation_tier=1):
        """
        Parameters
        ----------
        training_tier : int
            training tier index
        validation_tier : int
            validation tier index
        """
        from foundation.utility import standardize
        from foundation.stimulus import resize
        from foundation.recording import trial, trace, scan, tier, stat, resample
        from foundation.fnn.data import VisualScan, Data

        # filtered trials and traces
        scan.ScanTrials.populate(self.key, display_progress=True, reserve_jobs=True)
        self.unit_set.populate(self.key, display_progress=True, reserve_jobs=True)

        # trace orders
        self.unit_order.populate(self.key, display_progress=True, reserve_jobs=True)
        scan.ScanVisualPerspectiveOrder.populate(self.key, display_progress=True, reserve_jobs=True)
        scan.ScanVisualModulationOrder.populate(self.key, display_progress=True, reserve_jobs=True)

        for key in self.key:

            # data specification
            spec = fnn.Spec.VisualSpec & key

            def proj_spec(datatype):
                proj = {f"{_}_id": f"{_}_id_{datatype}" for _ in ["resample", "offset", "standardize"]}
                return spec.proj(..., **proj)

            # all trials
            all_trials = scan.ScanRecording & key
            all_trials = (trial.TrialSet & all_trials).members

            # filtered trials
            filt_trials = scan.ScanTrials & key

            # populate trials
            resample.TrialSamples.populate(all_trials, spec, display_progress=True, reserve_jobs=True)
            resample.ResampledTrial.populate(all_trials, spec, display_progress=True, reserve_jobs=True)
            tier.TrialTier.populate(filt_trials, key, display_progress=True, reserve_jobs=True)

            # videos
            videos = merge(all_trials, trial.TrialVideo)

            # populate videos
            resize.ResizedVideo.populate(videos, spec, display_progress=True, reserve_jobs=True)

            for table, datatype in [
                [recording.ScanVisualPerspectives, "perspective"],
                [recording.ScanVisualModulations, "modulation"],
                [self.unit_set, "unit"],
            ]:

                with cache_rowproperty():

                    # trace spec
                    _spec = proj_spec(datatype)

                    # traces
                    traceset = table & key
                    traces = (trace.TraceSet & traceset).members

                    # stats
                    stats = (standardize.Standardize & _spec).link.summary_ids
                    stats = [{"summary_id": _} for _ in stats]

                    # populate traces
                    stat.TraceSummary.populate(
                        traces, filt_trials, stats, _spec, display_progress=True, reserve_jobs=True
                    )
                    resample.ResampledTraces.populate(
                        traceset, all_trials, _spec, display_progress=True, reserve_jobs=True
                    )

            # insert
            key = dict(key, training_tier=training_tier, validation_tier=validation_tier)
            self.datatype.insert1(key, skip_duplicates=True)

        # fill
        Data.fill()


@keys
class VisualScanData(_VisualScanData):
    """Visual Scan Data -- Activity"""

    @property
    def keys(self):
        return [
            fnn.Spec.VisualSpec,
            pipe_fuse.ScanDone,
            recording.ScanVisualPerspectives,
            recording.ScanVisualModulations,
            recording.TraceFilterSet,
            recording.TrialFilterSet,
            recording.Tier,
        ]

    @property
    def unit_set(self):
        from foundation.recording.scan import ScanUnits

        return ScanUnits

    @property
    def unit_order(self):
        from foundation.recording.scan import ScanUnitOrder

        return ScanUnitOrder

    @property
    def datatype(self):
        return fnn.VisualScan


@keys
class VisualScanDataRaw(_VisualScanData):
    """Visual Scan Data -- Fluorescence"""

    @property
    def keys(self):
        return [
            fnn.Spec.VisualSpec,
            (scan.Scan * pipe_shared.PipelineVersion * pipe_shared.SegmentationMethod).proj()
            & pipe_fuse.ScanDone,
            recording.ScanVisualPerspectives,
            recording.ScanVisualModulations,
            recording.TraceFilterSet,
            recording.TrialFilterSet,
            recording.Tier,
        ]

    @property
    def unit_set(self):
        from foundation.recording.scan import ScanUnitsRaw

        return ScanUnitsRaw

    @property
    def unit_order(self):
        from foundation.recording.scan import ScanUnitRawOrder

        return ScanUnitRawOrder

    @property
    def datatype(self):
        return fnn.VisualScanRaw


@keys
class VisualScanIndividualModel:
    """Visual Scan Individual Model"""

    @property
    def keys(self):
        return [
            fnn.Data,
            fnn.Network,
            fnn.Instance.Individual,
        ]

    def fill(self):
        from foundation.fnn.model import Model

        for key in self.key:

            # instance parameters
            instance = (fnn.Instance.Individual & key).fetch1()
            instance.pop("instance_id")

            # train each cycle sequentially
            for cycle in range(instance["cycle"] + 1):

                # break if previous model cycle has not been trained
                if cycle and not Model & _key:
                    break

                # cycle instance
                _instance = dict(instance, cycle=cycle)
                _instance_id = (fnn.Instance.Individual & _instance).fetch1("instance_id")

                # populate model
                _key = dict(key, instance_id=_instance_id)
                Model.populate(_key, reserve_jobs=True)


@keys
class VisualScanFoundationModel:
    """Visual Scan Foundation Model"""

    @property
    def keys(self):
        return [
            fnn.Network,
            fnn.Instance.Foundation,
        ]

    def fill(self):
        from foundation.fnn.model import Model

        for key in self.key:

            # instance parameters
            instance = (fnn.Instance.Foundation & key).fetch1()
            instance.pop("instance_id")

            # train each cycle sequentially
            for cycle in range(instance["cycle"] + 1):

                # break if previous model cycle has not been trained
                if cycle and not Model & _key:
                    break

                # cycle instance
                _instance = dict(instance, cycle=cycle)
                _instance_id = (fnn.Instance.Foundation & _instance).fetch1("instance_id")

                # populate model
                _key = dict(key, instance_id=_instance_id)
                Model.populate(_key, reserve_jobs=True)


@keys
class VisualScanCorrelation:
    """Visual Scan Correlation"""

    @property
    def keys(self):
        return [
            (scan.Scan * fnn.Model) & fnn.Data.VisualScan,
            recording.TrialFilterSet,
            stimulus.VideoSet,
            utility.Correlation,
            utility.Burnin,
            utility.Bool.proj(perspective="bool"),
            utility.Bool.proj(modulation="bool"),
        ]

    def fill(self, cuda=True):
        from foundation.fnn.visual import VisualRecordingCorrelation
        from foundation.utils import use_cuda
        from contextlib import nullcontext

        # cuda context
        context = use_cuda if cuda else nullcontext

        with context():

            # correlation
            VisualRecordingCorrelation.populate(self.key, reserve_jobs=True, display_progress=True)


@keys
class VisualScanDirectionTuning:
    """Visual Scan Direction Tuning"""

    @property
    def keys(self):
        return [
            (scan.Scan * fnn.Model) & fnn.Data.VisualScan,
            stimulus.VideoSet,
            utility.Offset,
            utility.Impulse,
            utility.Precision,
            utility.Burnin,
        ]

    def fill(self, cuda=True):
        from foundation.fnn.visual import VisualDirectionTuning
        from foundation.utils import use_cuda
        from contextlib import nullcontext

        # cuda context
        context = use_cuda if cuda else nullcontext

        with context():

            # direction tuning
            VisualDirectionTuning.populate(self.key, reserve_jobs=True, display_progress=True)


@keys
class VisualScanSpatialTuning:
    """Visual Scan Spatial Tuning"""

    @property
    def keys(self):
        return [
            (scan.Scan * fnn.Model) & fnn.Data.VisualScan,
            stimulus.VideoSet,
            utility.Offset,
            utility.Impulse,
            utility.Resolution,
            utility.Burnin,
        ]

    def fill(self, cuda=True):
        from foundation.fnn.visual import VisualSpatialTuning
        from foundation.utils import use_cuda
        from contextlib import nullcontext

        # cuda context
        context = use_cuda if cuda else nullcontext

        with context():

            # direction tuning
            VisualSpatialTuning.populate(self.key, reserve_jobs=True, display_progress=True)
