from djutils import keys, merge, rowproperty
from foundation.virtual import utility, recording


# ----------------------------- Standardize -----------------------------


@keys
class StandardizedTrace:
    """Standardized Trace"""

    @property
    def keys(self):
        return [
            recording.Trace,
            recording.TrialSet & "members > 0",
            utility.Standardize,
            utility.Resample,
            utility.Offset,
            utility.Rate,
        ]

    @rowproperty
    def transform(self):
        """
        Returns
        -------
        foundation.utility.standardize.Standardize
            callable, standardizes trace
        """
        from foundation.utility.standardize import Standardize

        # homogeneous mask
        hom = merge(self.key, recording.TraceHomogeneous).fetch1("homogeneous")
        hom = [hom.astype(bool)]

        # standardization link
        stand = (Standardize & self.item).link

        # summary stats
        stat_keys = [{"summary_id": _} for _ in stand.summary_ids]
        stats = (utility.Summary & stat_keys).proj()
        stats = merge(self.key * stats, recording.TraceSummary)

        # stats dict
        summary_id, summary = stats.fetch("summary_id", "summary")
        kwargs = {k: [v] for k, v in zip(summary_id, summary)}

        # standarization transform
        return stand.standardize(homogeneous=hom, **kwargs)


@keys
class StandardizedTraces:
    """Standardized Trace Set"""

    @property
    def keys(self):
        return [
            recording.TraceSet & "members > 0",
            recording.TrialSet & "members > 0",
            utility.Standardize,
            utility.Resample,
            utility.Offset,
            utility.Rate,
        ]

    @rowproperty
    def transform(self):
        """
        Returns
        -------
        foundation.utility.standardize.Standardize
            callable, standardizes trace set
        """
        from foundation.utility.standardize import Standardize
        from foundation.recording.trace import TraceSet

        # traces
        traces = (TraceSet & self.item).members

        # homogeneous mask
        hom = merge(traces, recording.TraceHomogeneous)
        hom = hom.fetch("homogeneous", order_by="traceset_index")
        hom = hom.astype(bool)

        # standardization link
        stand = (Standardize & self.item).link

        # summary stats
        stat_keys = [{"summary_id": _} for _ in stand.summary_ids]
        stats = (utility.Summary & stat_keys).proj()
        stats = merge(self.key * traces * stats, recording.TraceSummary)

        # stats dict
        kwargs = dict()
        for skey in stat_keys:
            sid = skey["summary_id"]
            kwargs[sid] = (stats & skey).fetch("summary", order_by="traceset_index")

        # standarization transform
        return stand.standardize(homogeneous=hom, **kwargs)
