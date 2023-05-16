import pandas as pd
from djutils import rowproperty
from foundation.virtual.bridge import pipe_exp, pipe_shared
from foundation.virtual import utility, scan, recording
from foundation.schemas import fnn as schema


# -------------- Visual Spec --------------

# -- Visual Spec Base --


class _VisualSpec:
    """Visual Data Specification"""

    @rowproperty
    def data_keys(self):
        """
        Returns
        -------
        set[foundation.fnn.compute.LoadData]
            keys with `load` rowproperty
        """
        raise NotImplementedError()


# -- Visual Spec Types --


@schema.lookup
class ResampleVisual(_VisualSpec):
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
    def data_keys(self):
        from foundation.fnn.compute import ResampledVisualRecording

        return {ResampledVisualRecording}


# -- Data Spec --


@schema.link
class VisualSpec:
    links = [ResampleVisual]
    name = "visualspec"
    comment = "visual data specfication"
