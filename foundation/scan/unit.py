import datajoint as dj
from djutils import merge, row_method
from foundation.scan import timing
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
    -> timing.Timing
    -> UnitFilterSet
    ---
    -> UnitSet
    """

    @property
    def key_source(self):
        return timing.Timing.proj() * UnitFilterSet.proj() & pipe_fuse.ScanDone

    def make(self, key):
        # scan units
        units = pipe_fuse.ScanSet.Unit & key

        # filter units
        units = (UnitFilterSet & key).filter(units)

        # insert unit set
        unit_set = UnitSet.fill(units, prompt=False)

        # insert key
        self.insert1(dict(key, **unit_set))
