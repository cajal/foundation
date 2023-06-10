from datajoint import U
from djutils import merge, rowproperty
from foundation.virtual import utility, stimulus, recording
from foundation.schemas import fnn as schema


# ---------------------------- Spec ----------------------------

# -- Spec Types --


@schema.lookup
class VideoSpec:
    definition = """
    -> utility.Resize
    -> utility.Resolution
    """


@schema.lookup
class TraceSpec:
    definition = """
    -> utility.Standardize
    -> utility.Resample
    -> utility.Offset
    """


# -- Spec --


@schema.link
class Spec:
    links = [VideoSpec, TraceSpec]
    name = "spec"
    comment = "data specification"


# ---------------------------- Data ----------------------------


class _Data:
    """Data"""

    @rowproperty
    def network_data(self):
        """
        Returns
        -------
        foundation.fnn.compute_data.NetworkData (row)
            network input
        """
        raise NotImplementedError()

    @rowproperty
    def network_input(self):
        """
        Returns
        -------
        foundation.fnn.compute_input.NetworkInput (row)
            network input
        """
        raise NotImplementedError()


# -- Data Set Types --


@schema.lookup
class VisualScan(_Data):
    definition = """
    -> recording.ScanVisualPerspectives
    -> recording.ScanVisualModulations
    -> recording.ScanUnits
    -> recording.ScanTrials
    -> Spec.VideoSpec.proj(stimuli_id="spec_id")
    -> Spec.TraceSpec.proj(perspectives_id="spec_id")
    -> Spec.TraceSpec.proj(modulations_id="spec_id")
    -> Spec.TraceSpec.proj(units_id="spec_id")
    -> utility.Rate
    """

    @rowproperty
    def network_data(self):
        from foundation.fnn.compute_data import VisualScan

        return VisualScan & self

    @rowproperty
    def network_input(self):
        from foundation.fnn.compute_input import VisualScan

        return VisualScan & self


# -- Data Set Types --


@schema.link
class Data:
    links = [VisualScan]
    name = "data"
    comment = "fnn data"
