from djutils import keys, merge, cache_rowproperty
from foundation.virtual import recording, fnn


@keys
class VisualScanData:
    """Scan Data"""

    @property
    def keys(self):
        return [
            fnn.Spec.VisualSpec,
            recording.ScanVisualPerspectives,
            recording.ScanVisualModulations,
            recording.ScanUnits,
            recording.ScanTrials,
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
        from foundation.utility.standardize import Standardize
        from foundation.stimulus.resize import ResizedVideo
        from foundation.recording.tier import TrialTier
        from foundation.recording.trial import TrialSet, TrialVideo
        from foundation.recording.trace import TraceSet, TraceSummary
        from foundation.recording.resample import TrialSamples, ResampledTrial, ResampledTraces
        from foundation.fnn.data import VisualScan, Data

        for key in self.key:

            # data specification
            spec = fnn.Spec.VisualSpec & key

            def proj_spec(datatype):
                proj = {f"{_}_id": f"{_}_id_{datatype}" for _ in ["resample", "offset", "standardize"]}
                return spec.proj(..., **proj)

            # all trials
            all_trials = recording.ScanRecording & key
            all_trials = (TrialSet & all_trials).members

            # filtered trials
            filt_trials = recording.ScanTrials & key

            # populate trials
            TrialSamples.populate(all_trials, spec, display_progress=True, reserve_jobs=True)
            ResampledTrial.populate(all_trials, spec, display_progress=True, reserve_jobs=True)
            TrialTier.populate(filt_trials, key, display_progress=True, reserve_jobs=True)

            # videos
            videos = merge(all_trials, TrialVideo)

            # populate videos
            ResizedVideo.populate(videos, spec, display_progress=True, reserve_jobs=True)

            for table, proj in [
                [recording.ScanVisualPerspectives, "perspective"],
                [recording.ScanVisualModulations, "modulation"],
                [recording.ScanUnits, "unit"],
            ]:

                with cache_rowproperty():

                    # trace spec
                    _spec = proj_spec(proj)

                    # traces
                    traceset = table & key
                    traces = (TraceSet & traceset).members

                    # stats
                    stats = (Standardize & _spec).link.summary_ids
                    stats = [{"summary_id": _} for _ in stats]

                    # populate traces
                    TraceSummary.populate(traces, filt_trials, stats, _spec, display_progress=True, reserve_jobs=True)
                    ResampledTraces.populate(traceset, all_trials, _spec, display_progress=True, reserve_jobs=True)

            # insert
            key = dict(key, training_tier=training_tier, validation_tier=validation_tier)
            VisualScan.insert1(key, skip_duplicates=True)

        # fill
        Data.fill()
