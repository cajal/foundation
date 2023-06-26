from djutils import merge, rowproperty, U
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
    -> utility.Resample.proj(perspective_resample_id="resample_id")
    -> utility.Resample.proj(modulation_resample_id="resample_id")
    -> utility.Resample.proj(unit_resample_id="resample_id")
    -> utility.Offset.proj(perspective_offset_id="offset_id")
    -> utility.Offset.proj(modulation_offset_id="offset_id")
    -> utility.Offset.proj(unit_offset_id="offset_id")
    -> utility.Standardize.proj(perspective_standardize_id="standardize_id")
    -> utility.Standardize.proj(modulation_standardize_id="standardize_id")
    -> utility.Standardize.proj(unit_standardize_id="standardize_id")
    """


# -- Spec --


@schema.link
class Spec:
    links = [VisualSpec]
    name = "spec"
    comment = "data specification"


# ---------------------------- Data ----------------------------


class _Data:
    """Data"""

    @rowproperty
    def compute(self):
        """
        Returns
        -------
        foundation.fnn.compute_data.DataType (row)
            compute data
        """
        raise NotImplementedError()


# -- Data Set Types --


@schema.lookup
class VisualScan(_Data):
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
        from foundation.fnn.compute_data import VisualScan

        return VisualScan & self


# -- Data Set Types --


@schema.link
class Data:
    links = [VisualScan]
    name = "data"
    comment = "network data"
