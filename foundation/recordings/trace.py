import numpy as np
import datajoint as dj
from djutils import link, group, method, row_method, row_property, MissingError, RestrictionError
from foundation.utils.logging import logger
from foundation.utils.trace import truncate
from foundation.bridges.pipeline import pipe_meso, pipe_eye, pipe_tread
from foundation.recordings import trial

schema = dj.schema("foundation_recordings")


# -------------- Trace --------------

# -- Trace Base --


class TraceBase:
    """Recording Trace"""

    @row_property
    def times_values(self):
        """
        Returns
        -------
        times : 1D array
            trace times
        values : 1D array
            trace values, same length as times
        """
        raise NotImplementedError()

    @row_property
    def trials(self):
        """
        Returns
        -------
        trials.Trial
            tuples from trials.Trial
        """
        raise NotImplementedError()


# -- Trace Types --


class ScanBase(TraceBase):
    """Scan Trials"""

    @row_property
    def scan_key(self):
        key = ["animal_id", "session", "scan_idx"]
        return dict(zip(key, self.fetch1(*key)))

    @row_property
    def trials(self):
        from foundation.bridges.pipeline import pipe_stim

        scan_trials = pipe_stim.Trial & self
        keys = (trial.TrialLink.ScanTrial * scan_trials).proj()

        if scan_trials - keys:
            raise MissingError("Missing trials.")

        if keys - scan_trials:
            raise RestrictionError("Unexpected trials.")

        return trial.TrialLink & keys


@schema
class MesoActivity(ScanBase, dj.Lookup):
    definition = """
    -> pipe_meso.Activity.Trace
    """

    @row_property
    def times_values(self):
        from foundation.recordings.scan import stimulus_times

        # scan times on stimulus clock
        times = stimulus_times(**self.scan_key)

        # imaging delay
        delay = (pipe_meso.ScanSet.UnitInfo & self).fetch1("ms_delay") / 1000

        # activity trace
        values = (pipe_meso.Activity.Trace & self).fetch1("trace")

        # trim to same length
        times, values = truncate(times, values, tolerance=1)

        return times + delay, values


@schema
class ScanPupilType(dj.Lookup):
    definition = """
    pupil_type          : varchar(64)   # fitted scan pupil type
    pupil_attribute     : varchar(64)   # fitted scan pupil attribute
    """


@schema
class ScanPupil(ScanBase, dj.Lookup):
    definition = """
    -> pipe_eye.FittedPupil
    -> ScanPupilType
    """

    @row_property
    def times_values(self):
        from foundation.recordings.scan import eye_times

        # eye times on stimulus clock
        times = eye_times(**self.scan_key)

        # fetch trace based on pupil type and attribute
        pupil_type, pupil_attr = self.fetch1("pupil_type", "pupil_attribute")

        if pupil_type == "circle":
            # fitted pupil circle
            fits = pipe_eye.FittedPupil.Circle & self

            if pupil_attr == "radius":
                # fitted circle radius
                values = fits.fetch("radius", order_by="frame_id")

            elif pupil_attr in ["center_x", "center_y"]:
                # fitted circle center
                center = fits.fetch("center", order_by="frame_id")

                if pupil_attr == "center_x":
                    values = np.array([np.nan if c is None else c[0] for c in center])
                else:
                    values = np.array([np.nan if c is None else c[1] for c in center])

            else:
                # other fitted circle attributes not implemented
                raise NotImplementedError()

        else:
            # other types not implemented
            raise NotImplementedError()

        return times, values


@schema
class ScanTreadmill(ScanBase, dj.Lookup):
    definition = """
    -> pipe_tread.Treadmill
    """

    @row_property
    def times_values(self):
        from foundation.recordings.scan import treadmill_times

        times = treadmill_times(**self.scan_key)
        values = (pipe_tread.Treadmill & self).fetch1("treadmill_vel")

        return times, values


# -- Trace Link --


@link(schema)
class TraceLink:
    links = [MesoActivity, ScanPupil, ScanTreadmill]
    name = "trace"
    comment = "recording trace"
