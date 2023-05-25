from djutils import rowproperty
from foundation.virtual.bridge import pipe_exp, pipe_shared
from foundation.virtual import utility, recording
from foundation.schemas import fnn as schema


# -------------- Data Spec --------------

# -- Data Spec Types --


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


# -- Data Spec --


@schema.link
class Spec:
    links = [VideoSpec, TraceSpec]
    name = "spec"
    comment = "data specification"


# -------------- Data Set --------------


class _DataSet:
    """Data Set"""

    @rowproperty
    def trainset(self):
        """
        Returns
        -------
        fnn.data.Dataset
            training dataset
        """
        raise NotImplementedError()

    @rowproperty
    def network_sizes(self):
        """
        Returns
        -------
        dict
            network sizes
        """
        raise NotImplementedError()

    @rowproperty
    def visual_inputs(self):
        """
        Returns
        -------
        djutils.derived.Keys
            key_list -- [foundation.stimulus.Video, ...]
            rowmethod -- [trials, stimuli, perspectives, modulations, ...]
        """
        raise NotImplementedError()

    @rowproperty
    def response_timing(self):
        """
        Returns
        -------
        float
            response period (seconds)
        float
            response offset (seconds)
        """
        raise NotImplementedError()


# -- Data Set Types --


@schema.lookup
class VisualScan(_DataSet):
    definition = """
    -> pipe_exp.Scan
    -> pipe_shared.TrackingMethod
    -> pipe_shared.SpikeMethod
    -> recording.ScanTrials
    -> recording.ScanPerspectives
    -> recording.ScanModulations
    -> recording.ScanUnits
    -> DataSpec.VideoSpec.proj(stimuli_id="spec_id")
    -> DataSpec.TraceSpec.proj(perspectives_id="spec_id")
    -> DataSpec.TraceSpec.proj(modulations_id="spec_id")
    -> DataSpec.TraceSpec.proj(units_id="spec_id")
    -> utility.Rate
    """

    @rowproperty
    def trainset(self):
        from foundation.fnn.compute import VisualScan

        return (VisualScan & self).trainset

    @rowproperty
    def network_sizes(self):
        from foundation.fnn.compute import VisualScan

        return (VisualScan & self).network_sizes

    @rowproperty
    def visual_inputs(self):
        from foundation.fnn.compute import VisualScanInputs

        return VisualScanInputs & self

    @rowproperty
    def response_timing(self):
        from foundation.utility.resample import Rate, Offset

        key = self.proj(spec_id="units_id") * Spec.TraceSpec
        period = (Rate & key).link.period
        offset = (Offset & key).link.offset

        return period, offset


# -- Data Set Types --


@schema.link
class Data:
    links = [VisualScan]
    name = "data"
    comment = "network data"
