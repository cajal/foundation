import numpy as np
import datajoint as dj
from djutils import link, method, row_method
from tqdm import tqdm
from foundation.utils.logging import logger
from foundation.recordings import trial, trace, resample


schema = dj.schema("foundation_recordings")


# -------------- Nan --------------

# -- Nan Base --


class NansBase:
    """Trace Nans"""

    @row_method
    def nans(self, times, values):
        """
        Returns
        -------
        foundation.utils.nan.Nans
            detects nans
        """
        raise NotImplementedError()


# -- Nans Types --


@schema
class DurationNans(dj.Lookup):
    definition = """
    reduce      : varchar(32)   # reduction method
    """

    @row_method
    def nans(self, times, values):
        from foundation.utils.nans import DurationNans

        return DurationNans(
            times=times,
            values=values,
            reduce=self.fetch1("reduce"),
        )


# -- Nans Link --


@link(schema)
class NansLink:
    links = [DurationNans]
    name = "nans"
    comment = "nans method"


# -- Computed Nans --


@schema
class TrialNans(dj.Computed):
    definition = """
    -> trial.TrialLink
    -> trace.TraceLink
    -> resample.OffsetLink
    -> NansLink
    ---
    nans = NULL         : float     # detected nans
    """

    @property
    def key_source(self):
        return trace.TraceLink.proj() * resample.OffsetLink().proj() * NansLink.proj()

    def make(self, key):
        # trace link
        trace_link = (trace.TraceLink & key).link

        # detect nans in trace
        times, values = trace_link.times_values
        nans = (NansLink & key).link.nans(times, values)

        # trace trials
        trials = trace_link.trials
        trial_keys = trials.fetch(dj.key, order_by=trials.primary_key)

        # offset (seconds)
        offset = (resample.OffsetLink & key).link.offset

        keys = []
        for trial_key in tqdm(trial_keys):

            # flip times
            flips = (trial.TrialLink & trial_key).link.flips
            start = flips.min() + offset
            end = flips.max() + offset

            # detect nans
            trial_key = dict(nans=nans(start, end), **key, **trial_key)
            keys.append(trial_key)

        self.insert(keys)
