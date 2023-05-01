import datajoint as dj
from djutils import link, group, merge, row_method, skip_missing
from foundation.scan import timing
from foundation.schemas.pipeline import pipe_fuse, pipe_shared, resolve_pipe
from foundation.schemas import scan as schema


# -------------- Unit Set --------------


@group(schema)
class UnitSet:
    keys = [pipe_fuse.ScanSet.Unit]
    name = "units"
    comment = "set of scan units"


# -------------- Unit Filter --------------

# -- Unit Filter Base --


class UnitFilterBase:
    """Unit Filter"""

    @row_method
    def filter(self, units):
        """
        Parameters
        ----------
        units : pipe_fuse.ScanSet.Unit
            tuples from pipe_fuse.ScanSet.Unit

        Returns
        -------
        pipe_fuse.ScanSet.Unit
            restricted tuples
        """
        raise NotImplementedError()


# -- Unit Filter Types --


@schema
class UnitMaskType(UnitFilterBase, dj.Lookup):
    definition = """
    -> pipe_shared.PipelineVersion
    -> pipe_shared.SegmentationMethod
    -> pipe_shared.ClassificationMethod
    -> pipe_shared.MaskType
    """

    @row_method
    def filter(self, units):
        key = dj.U("animal_id", "session", "scan_idx") & units
        pipe = resolve_pipe(**key.fetch1())

        key = merge(
            units,
            self.proj(target="type"),
            pipe.MaskClassification.Type * pipe.ScanSet.Unit,
        )

        return units & (key & "type = target")


# -- Unit Filter Link --


@link(schema)
class UnitFilterLink:
    links = [UnitMaskType]
    name = "unit_filter"
    comment = "scan unit filter"


@group(schema)
class UnitFilterSet:
    keys = [UnitFilterLink]
    name = "unit_filters"
    comment = "set of scan unit filters"


# -- Computed Unit Filter --


@schema
class FilteredUnits(dj.Computed):
    definition = """
    -> timing.Timing
    -> UnitFilterSet
    ---
    -> UnitSet
    """

    @property
    def key_source(self):
        return timing.Timing.proj() * UnitFilterSet.proj() & pipe_fuse.ScanDone

    @skip_missing
    def make(self, key):
        # scan units
        units = pipe_fuse.ScanSet.Unit & key

        # filter units
        for filter_key in (UnitFilterSet & key).members.fetch(dj.key, order_by="member_id"):
            units = (UnitFilterLink & key).link.filter(units)

        # insert unit set
        unit_set = UnitSet.fill(units, prompt=False)

        # insert key
        self.insert1(dict(key, **unit_set))
