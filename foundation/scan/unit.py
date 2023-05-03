import datajoint as dj
from djutils import merge, row_method
from foundation.scan import experiment
from foundation.schemas.pipeline import pipe_fuse, pipe_shared, resolve_pipe
from foundation.schemas import scan as schema


# -------------- Unit Set --------------


@schema.set
class UnitSet:
    keys = [pipe_fuse.ScanSet.Unit]
    name = "units"
    comment = "set of scan units"


# -------------- Unit Filter --------------


@schema.filter_lookup
class UnitMaskType:
    filter_type = pipe_fuse.ScanSet.Unit
    definition = """
    -> pipe_shared.PipelineVersion
    -> pipe_shared.SegmentationMethod
    -> pipe_shared.ClassificationMethod
    -> pipe_shared.MaskType
    """

    @row_method
    def filter(self, units):
        pipe = resolve_pipe(units)
        key = merge(
            units,
            self.proj(target="type"),
            pipe.MaskClassification.Type * pipe.ScanSet.Unit,
        )
        return units & (key & "type = target")


@schema.filter_link
class UnitFilterLink:
    filters = [UnitMaskType]
    name = "unit_filter"
    comment = "scan unit filter"


@schema.filter_link_set
class UnitFilterSet:
    filter_link = UnitFilterLink
    name = "unit_filters"
    comment = "set of scan unit filters"


# -- Computed Unit Filter --


@schema.computed
class FilteredUnits:
    definition = """
    -> experiment.Scan
    -> UnitFilterSet
    ---
    -> UnitSet
    """

    @property
    def key_source(self):
        return experiment.Scan.proj() * UnitFilterSet.proj() & pipe_fuse.ScanDone

    def make(self, key):
        # scan units
        units = pipe_fuse.ScanSet.Unit & key

        # filter units
        units = (UnitFilterSet & key).filter(units)

        # insert unit set
        unit_set = UnitSet.fill(units, prompt=False)

        # insert key
        self.insert1(dict(key, **unit_set))

    def fill_units(self, spike_method):
        """
        Parameters
        ----------
        spike_method : pipe_shared.SpikeMethod
            tuple(s)
        """
        from foundation.recording.trace import ScanUnit, TraceLink, TraceHomogeneous, TraceTrials

        # scan unit traces
        units = UnitSet.Member & self
        units = units * spike_method.proj()
        ScanUnit.insert(units, skip_duplicates=True, ignore_extra_fields=True)

        # trace link
        TraceLink.fill()

        # compute trace
        key = TraceLink.ScanUnit & units
        TraceHomogeneous.populate(key, display_progress=True, reserve_jobs=True)
        TraceTrials.populate(key, display_progress=True, reserve_jobs=True)
