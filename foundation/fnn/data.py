import pandas as pd
from djutils import rowproperty
from foundation.virtual.bridge import pipe_exp, pipe_shared
from foundation.virtual import utility, scan, recording
from foundation.schemas import fnn as schema


# -------------- Data Set --------------

# -- Data Set Base --


class _DataSet:
    """Data Set"""

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


# -- Data Set Types --


@schema.lookup
class Scan(_DataSet):
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


# -- Data Set --


@schema.link
class DataSet:
    links = [Scan]
    name = "dataset"


@schema.linkset
class DataSets:
    link = DataSet
    name = "datasets"


# -- Computed Data Set --


@schema.computed
class DataSetComponents:
    definition = """
    -> DataSet
    ---
    -> recording.TrialSet
    -> recording.TraceSet.proj(traceset_id_p="traceset_id")
    -> recording.TraceSet.proj(traceset_id_m="traceset_id")
    -> recording.TraceSet.proj(traceset_id_u="traceset_id")
    """

    def make(self, key):
        dataset = (DataSet & key).link

        key["trialset_id"] = dataset.trial_set.fetch1("trialset_id")
        key["traceset_id_p"] = dataset.perpsective_set.fetch1("traceset_id")
        key["traceset_id_m"] = dataset.modulation_set.fetch1("traceset_id")
        key["traceset_id_u"] = dataset.unit_set.fetch1("traceset_id")

        self.insert1(key)


# -------------- Data Spec --------------

# -- Data Spec Base --


class _DataSpec:
    """Data Specification"""

    @rowproperty
    def loader(self):
        """
        Returns
        -------
        foundation.fnn.compute.LoadData
            keys with `load` rowproperty
        """
        raise NotImplementedError()


# -- Data Spec Types --


@schema.lookup
class Preprocess(_DataSpec):
    definition = """
    -> utility.Resolution
    -> utility.Resize
    -> utility.Rate
    -> utility.Offset.proj(offset_id_p="offset_id")
    -> utility.Offset.proj(offset_id_m="offset_id")
    -> utility.Offset.proj(offset_id_u="offset_id")
    -> utility.Resample.proj(resample_id_p="resample_id")
    -> utility.Resample.proj(resample_id_m="resample_id")
    -> utility.Resample.proj(resample_id_u="resample_id")
    -> utility.Standardize.proj(standardize_id_p="standardize_id")
    -> utility.Standardize.proj(standardize_id_m="standardize_id")
    -> utility.Standardize.proj(standardize_id_u="standardize_id")
    """

    @rowproperty
    def loader(self):
        from foundation.fnn.compute import PreprocessedData

        return PreprocessedData


# -- Data Spec --


@schema.link
class DataSpec:
    links = [Preprocess]
    name = "dataspec"
