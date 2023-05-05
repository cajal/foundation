from djutils import keys, merge, row_property
from foundation.utility.standardize import StandardizeLink
from foundation.utility.resample import RateLink, OffsetLink, ResampleLink
from foundation.recording.trial import TrialSet
from foundation.recording.trace import TraceSet, TraceHomogeneous
from foundation.recording.stat import TraceSummary


@keys
class StandardizeTraces:
    keys = [TraceSet, TrialSet, RateLink, OffsetLink, ResampleLink, StandardizeLink]

    @row_property
    def transform(self):
        # trace and stat keys
        trace_keys = (TraceSet & self.key).members
        stat_keys = (StandardizeLink & self.key).link.summary_keys

        # homogeneous mask
        hom = merge(trace_keys, TraceHomogeneous)
        hom = hom.fetch("homogeneous", order_by="trace_id ASC")
        hom = hom.astype(bool)

        # summary stats
        keys = trace_keys * self.key * stat_keys
        keys = merge(keys, TraceSummary)

        stats = dict()
        for summary_id, df in keys.fetch(format="frame").groupby("summary_id"):

            df = df.sort_values("trace_id", ascending=True)
            stats[summary_id] = df.summary.values

        # standarization transform
        return (StandardizeLink & self.key).link.standardize(homogeneous=hom, **stats)
