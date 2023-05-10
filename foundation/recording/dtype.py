from djutils import rowproperty
from foundation.virtual.bridge import pipe_shared
from foundation.virtual import utility
from foundation.recording.trial import Trial
from foundation.recording.trace import TraceFilterSet
from foundation.recording.scan import ScanTrials, ScanPerspectives, ScanModulations, ScanUnits
from foundation.schemas import recording as schema


# -------------- Dtype --------------

# -- Dtype Base --


class _Dtype:
    """Recording Data Type"""

    @rowproperty
    def data(self):
        """
        Returns
        -------
        foundation.recording.load.Data
            data loader
        """
        raise NotImplementedError()


# -- Dtype Types --


@schema.lookup
class VideoStimulus(_Dtype):
    definition = """
    -> utility.Resize
    -> utility.Resolution
    """

    @rowproperty
    def data(self):
        from foundation.recording.load import Video

        return Video & self


@schema.lookup
class ScanPerspectiveTraces(_Dtype):
    definition = """
    -> pipe_shared.TrackingMethod
    -> TraceFilterSet
    -> utility.Standardize
    -> utility.Resample
    -> utility.Offset
    """

    @rowproperty
    def data(self):
        from foundation.recording.load import Traces

        return Traces & (self * ScanPerspectives * ScanTrials * Trial.ScanTrial)


@schema.lookup
class ScanModulationTraces(_Dtype):
    definition = """
    -> pipe_shared.TrackingMethod
    -> TraceFilterSet
    -> utility.Standardize
    -> utility.Resample
    -> utility.Offset
    """

    @rowproperty
    def data(self):
        from foundation.recording.load import Traces

        return Traces & (self * ScanModulations * ScanTrials * Trial.ScanTrial)


@schema.lookup
class ScanUnitTraces(_Dtype):
    definition = """
    -> pipe_shared.SpikeMethod
    -> TraceFilterSet
    -> utility.Standardize
    -> utility.Resample
    -> utility.Offset
    """

    @rowproperty
    def data(self):
        from foundation.recording.load import Traces

        return Traces & (self * ScanUnits * ScanTrials * Trial.ScanTrial)


# -- Dtype --


@schema.link
class Dtype:
    links = [VideoStimulus, ScanPerspectiveTraces, ScanModulationTraces, ScanUnitTraces]
    name = "dtype"
    comment = "data type"
