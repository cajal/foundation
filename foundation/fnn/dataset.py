import pandas as pd
from djutils import rowproperty
from foundation.virtual.bridge import pipe_exp, pipe_shared
from foundation.virtual import utility, scan, recording
from foundation.schemas import fnn as schema


# -------------- Visual Set --------------

# -- Visual Set Base --


class _VisualSet:
    """Visual Set"""

    @rowproperty
    def trial_set(self):
        """
        Returns
        -------
        foundation.virtual.recording.TrialSet
            tuple
        """
        raise NotImplementedError()

    @rowproperty
    def perpsective_set(self):
        """
        Returns
        -------
        foundation.virtual.recording.TraceSet
            tuple
        """
        raise NotImplementedError()

    @rowproperty
    def modulation_set(self):
        """
        Returns
        -------
        foundation.virtual.recording.TraceSet
            tuple
        """
        raise NotImplementedError()

    @rowproperty
    def unit_set(self):
        """
        Returns
        -------
        foundation.virtual.recording.TraceSet
            tuple
        """
        raise NotImplementedError()

    @rowproperty
    def datakeys(self):
        """
        Returns
        -------
        set[djutils.derived.Keys]
            keys with `dataset` rowproperty
        """
        raise NotImplementedError()


# -- Visual Set Types --


@schema.lookup
class VisualScan(_VisualSet):
    definition = """
    -> pipe_exp.Scan
    -> pipe_shared.TrackingMethod
    -> pipe_shared.SpikeMethod
    -> recording.ScanTrials
    -> recording.ScanPerspectives
    -> recording.ScanModulations
    -> recording.ScanUnits
    """

    @rowproperty
    def trial_set(self):
        return recording.TrialSet & (recording.ScanTrials & self)

    @rowproperty
    def perpsective_set(self):
        return recording.TraceSet & (recording.ScanPerspectives & self)

    @rowproperty
    def modulation_set(self):
        return recording.TraceSet & (recording.ScanModulations & self)

    @rowproperty
    def unit_set(self):
        return recording.TraceSet & (recording.ScanUnits & self)

    @rowproperty
    def datakeys(self):
        from foundation.fnn.compute import ResampledVisualRecording

        return {ResampledVisualRecording}


# -- Data Set --


@schema.link
class VisualSet:
    links = [VisualScan]
    name = "visualset"
    comment = "visual data set"


# -- Computed Data Set --


@schema.computed
class VisualRecording:
    definition = """
    -> VisualSet
    ---
    -> recording.TrialSet
    -> recording.TraceSet.proj(traceset_id_p="traceset_id")
    -> recording.TraceSet.proj(traceset_id_m="traceset_id")
    -> recording.TraceSet.proj(traceset_id_u="traceset_id")
    """

    def make(self, key):
        dataset = (VisualSet & key).link

        key["trialset_id"] = dataset.trial_set.fetch1("trialset_id")
        key["traceset_id_p"] = dataset.perpsective_set.fetch1("traceset_id")
        key["traceset_id_m"] = dataset.modulation_set.fetch1("traceset_id")
        key["traceset_id_u"] = dataset.unit_set.fetch1("traceset_id")

        self.insert1(key)
