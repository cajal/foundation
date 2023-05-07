import datajoint as dj
from djutils import merge, rowmethod
from foundation.scan.experiment import Scan
from foundation.virtual.bridge import pipe_fuse, pipe_shared, resolve_pipe
from foundation.schemas import scan as schema


# -------------- Unit Set --------------


@schema.set
class UnitSet:
    keys = [pipe_fuse.ScanSet.Unit]
    name = "units"
    comment = "scan unit set"
    part_name = "unit"


# -------------- Unit Filter --------------

# -- Filter Types --


@schema.filter_lookup
class UnitMaskType:
    ftype = pipe_fuse.ScanSet.Unit
    definition = """
    -> pipe_shared.PipelineVersion
    -> pipe_shared.SegmentationMethod
    -> pipe_shared.ClassificationMethod
    -> pipe_shared.MaskType
    """

    @rowmethod
    def filter(self, units):
        pipe = resolve_pipe(units)
        key = merge(
            units,
            self.proj(target="type"),
            pipe.MaskClassification.Type * pipe.ScanSet.Unit,
        )
        return units & (key & "type = target")


# -- Filter --


@schema.filter_link
class UnitFilterLink:
    links = [UnitMaskType]
    name = "unit_filter"
    comment = "scan unit filter"


# -- Filter Set --


@schema.filter_link_set
class UnitFilterSet:
    link = UnitFilterLink
    name = "unit_filters"
    comment = "scan unit filter set"


# -- Computed Filter --


@schema.computed
class FilteredUnits:
    definition = """
    -> Scan
    -> UnitFilterSet
    ---
    -> UnitSet
    """

    @property
    def key_source(self):
        return Scan.proj() * UnitFilterSet.proj() & pipe_fuse.ScanDone

    def make(self, key):
        # scan units
        units = pipe_fuse.ScanSet.Unit & key

        # filter units
        units = (UnitFilterSet & key).filter(units)

        # insert unit set
        unit_set = UnitSet.fill(units, prompt=False)

        # insert key
        self.insert1(dict(key, **unit_set))
