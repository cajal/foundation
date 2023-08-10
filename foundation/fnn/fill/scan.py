from djutils import keys, merge, cache_rowproperty
from foundation.virtual.bridge import pipe_fuse
from foundation.virtual import utility, stimulus, scan, recording, fnn


@keys
class VisualScanData:
    """Visual Scan Data"""

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
        scan.ScanUnits.populate(self.key, display_progress=True, reserve_jobs=True)

        # trace orders
        scan.ScanUnitOrder.populate(self.key, display_progress=True, reserve_jobs=True)
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
                [recording.ScanUnits, "unit"],
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
            VisualScan.insert1(key, skip_duplicates=True)

        # fill
        Data.fill()


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
class VisualScanCCNorm:
    """Visual Scan Normalized Correlation Coefficient"""

    @property
    def keys(self):
        return [
            (scan.Scan * fnn.Model) & fnn.Data.VisualScan,
            recording.TrialFilterSet,
            stimulus.VideoSet,
            utility.Burnin,
            utility.Bool.proj(perspective="bool"),
            utility.Bool.proj(modulation="bool"),
        ]

    def fill(self, cuda=True):
        from foundation.fnn.data import Data
        from foundation.fnn.visual import VisualRecordingCorrelation
        from foundation.recording.visual import VisualMeasure
        from foundation.recording.trace import TraceSet
        from foundation.utils import use_cuda
        from contextlib import nullcontext

        # cuda context
        context = use_cuda if cuda else nullcontext

        with context():

            # unit correlations
            VisualRecordingCorrelation.populate(
                self.key, utility.Correlation.CCSignal, reserve_jobs=True, display_progress=True
            )

        for key in self.key.fetch("data_id", "trial_filterset_id", "videoset_id", "burnin", as_dict=True):

            # unit key
            _key = (Data & key).link.compute.key_unit
            _key.pop("trial_filterset_id")

            # unit traces
            traces = (TraceSet & _key).members

            with cache_rowproperty():

                # unit measures
                VisualMeasure.populate(
                    key, _key, traces, utility.Measure.CCMax, reserve_jobs=True, display_progress=True
                )
