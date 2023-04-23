import numpy as np
import datajoint as dj
from djutils import link, group, method, row_method, row_property, MissingError, RestrictionError
from foundation.utils.logging import logger
from foundation.bridge.pipeline import pipe_meso, pipe_eye, pipe_tread
from foundation.recording import trial

schema = dj.schema("foundation_recording")


# -------------- Trace --------------

# -- Trace Base --


class TraceBase:
    """Recording Trace"""

    @row_property
    def times(self):
        """
        Returns
        -------
        times : 1D array
            trace times
        """
        raise NotImplementedError()

    @row_property
    def values(self):
        """
        Returns
        -------
        values : 1D array
            trace values
        """
        raise NotImplementedError()

    @row_property
    def trial_flips(self):
        """
        Returns
        -------
        trials.TrialFlips
            tuples from trials.TrialFlips
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
    def trial_flips(self):
        from foundation.bridge.pipeline import pipe_stim

        scan_trials = pipe_stim.Trial.proj() & self
        keys = trial.TrialFlips.proj() * trial.TrialLink.ScanTrial * scan_trials

        if scan_trials - keys:
            raise MissingError("Missing trials.")

        if keys - scan_trials:
            raise RestrictionError("Unexpected trials.")

        return trial.TrialFlips & keys


@schema
class MesoActivity(ScanBase, dj.Lookup):
    definition = """
    -> pipe_meso.Activity.Trace
    """

    @row_property
    def times(self):
        """
        Returns
        -------
        times : 1D array
            trace times
        """
        from foundation.recording.scan import scan_times

        times = scan_times(**self.scan_key)[0]
        delay = (pipe_meso.ScanSet.UnitInfo & self).fetch1("ms_delay") / 1000
        return times + delay

    @row_property
    def values(self):
        """
        Returns
        -------
        values : 1D array
            trace values
        """
        return (pipe_meso.Activity.Trace & self).fetch1("trace")


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
    def times(self):
        from foundation.recording.scan import eye_times

        return eye_times(**self.scan_key)

    @row_property
    def values(self):
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
                    return np.array([np.nan if c is None else c[0] for c in center])
                else:
                    return np.array([np.nan if c is None else c[1] for c in center])

            else:
                # other fitted circle attributes not implemented
                raise NotImplementedError()

        else:
            # other types not implemented
            raise NotImplementedError()


@schema
class ScanTreadmill(ScanBase, dj.Lookup):
    definition = """
    -> pipe_tread.Treadmill
    """

    @row_property
    def times(self):
        from foundation.recording.scan import treadmill_times

        return treadmill_times(**self.scan_key)

    @row_property
    def values(self):
        return (pipe_tread.Treadmill & self).fetch1("treadmill_vel")


# -- Trace Link --


@link(schema)
class TraceLink:
    links = [MesoActivity, ScanPupil, ScanTreadmill]
    name = "trace"
    comment = "recording trace"
