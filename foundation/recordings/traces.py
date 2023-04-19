import numpy as np
import datajoint as dj
from djutils import link, group, method, row_method, row_property, MissingError
from foundation.utils.logging import logger
from foundation.utils.traces import truncate
from foundation.recordings import trials

pipe_meso = dj.create_virtual_module("pipe_meso", "pipeline_meso")
schema = dj.schema("foundation_recordings")


# -------------- Trace --------------

# -- Base --


class TraceBase:
    """Recording Trace"""

    @row_property
    def trace_times(self):
        """
        Returns
        -------
        1D array
            recording trace
        1D array
            recording times of each point in the trace

        IMPORTANT : arrays must be the same length
        """
        raise NotImplementedError()

    @row_property
    def trial_times(self):
        """
        Returns
        -------
        Iterator[tuple[trials.Trial, 1D array]]
            yields
                trial.Trial
                    recording trial
                1D array
                    recording times of each stimulus flips
        """
        raise NotImplementedError()


# -- Types --


@schema
class MesoActivity(TraceBase, dj.Lookup):
    definition = """
    -> pipe_meso.Activity.Trace
    """

    @row_property
    def trace_times(self):
        from foundation.recordings.scan import planes

        # scan key
        scan_key = ["animal_id", "session", "scan_idx"]
        scan_key = dict(zip(scan_key, self.fetch1(*scan_key)))

        # number of scan planes
        n = planes(**scan_key)

        # start of each scan volume in stimulus clock
        stim = dj.create_virtual_module("stim", "pipeline_stimulus")
        times = (stim.Sync & scan_key).fetch1("frame_times")[::n]

        # verify times are all finite
        assert np.isfinite(times).all()

        # activity trace
        trace = (pipe_meso.Activity.Trace & self).fetch1("trace")

        # trim to same length
        trace, times = truncate(trace, times, tolerance=1)

        # imaging delay
        delay = (pipe_meso.ScanSet.UnitInfo & self).fetch1("ms_delay") / 1000

        return trace, times + delay

    @row_property
    def trial_times(self):

        # restrict trials
        key = trials.TrialsLink.ScanTrials * trials.ScanTrials & self
        keys = (trials.Trials & key).trials

        # iterate through trials
        for key in keys.fetch(dj.key, order_by=keys.primary_key):

            trial = trials.Trial & key
            flips = trial.flips

            yield trial, flips
