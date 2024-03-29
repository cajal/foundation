from djutils import merge, rowproperty
from foundation.virtual import utility, stimulus, recording
from foundation.schemas import fnn as schema


# ---------------------------- Spec ----------------------------

# -- Spec Types --


@schema.lookup
class VisualSpec:
    definition = """
    -> utility.Resolution
    -> utility.Resize
    -> utility.Rate
    -> utility.Resample.proj(resample_id_perspective="resample_id")
    -> utility.Resample.proj(resample_id_modulation="resample_id")
    -> utility.Resample.proj(resample_id_unit="resample_id")
    -> utility.Offset.proj(offset_id_perspective="offset_id")
    -> utility.Offset.proj(offset_id_modulation="offset_id")
    -> utility.Offset.proj(offset_id_unit="offset_id")
    -> utility.Standardize.proj(standardize_id_perspective="standardize_id")
    -> utility.Standardize.proj(standardize_id_modulation="standardize_id")
    -> utility.Standardize.proj(standardize_id_unit="standardize_id")
    """


# -- Spec --


@schema.link
class Spec:
    links = [VisualSpec]
    name = "spec"
    comment = "data specification"


# ---------------------------- Data ----------------------------

# -- Data Interface --


class DataType:
    """Data"""

    @rowproperty
    def compute(self):
        """
        Returns
        -------
        foundation.fnn.compute.data.DataType (row)
            compute data
        """
        raise NotImplementedError()


# -- Data Types --


@schema.lookup
class VisualScan(DataType):
    definition = """
    -> Spec.VisualSpec
    -> recording.ScanVisualPerspectives
    -> recording.ScanVisualModulations
    -> recording.ScanUnits
    -> recording.ScanTrials
    -> recording.Tier
    training_tier       : int unsigned  # training tier index
    validation_tier     : int unsigned  # validation tier index
    """

    @rowproperty
    def compute(self):
        from foundation.fnn.compute.data import VisualScan

        return VisualScan & self


@schema.lookup
class VisualScanRaw(DataType):
    definition = """
    -> Spec.VisualSpec
    -> recording.ScanVisualPerspectives
    -> recording.ScanVisualModulations
    -> recording.ScanUnitsRaw
    -> recording.ScanTrials
    -> recording.Tier
    training_tier       : int unsigned  # training tier index
    validation_tier     : int unsigned  # validation tier index
    """

    @rowproperty
    def compute(self):
        from foundation.fnn.compute.data import VisualScanRaw

        return VisualScanRaw & self


@schema.lookup
class Sensorium2023(DataType):
    definition = """
    sensorium_dataset   : varchar(512)  # sensorium dataset
    trainval_tiers      : varchar(512)  # training & validation tiers (csv)
    split_fraction      : decimal(6, 6) # split fraction
    split_seed          : int unsigned  # split seed
    """


# -- Data --


@schema.link
class Data:
    links = [VisualScan, VisualScanRaw]
    name = "data"
    comment = "fnn data"


@schema.linkset
class DataSet:
    link = Data
    name = "dataset"
    comment = "fnn data set"
