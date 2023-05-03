import numpy as np
from foundation.scan import experiment
from foundation.schemas.pipeline import pipe_eye, pipe_stim
from foundation.schemas import scan as schema


@schema.computed
class PupilTrace:
    definition = """
    -> experiment.Scan
    -> pipe_eye.FittedPupil
    pupil_type      : enum("radius", "center_x", "center_y")    # pupil data type
    ---
    pupil_trace     : longblob  # pupil trace
    """

    def make(self, key):
        # fetch pupil fits
        fits = pipe_eye.FittedPupil.Circle & key
        radius, center = fits.fetch("radius", "center", order_by="frame_id")

        # fill nans
        center = [np.array([np.nan, np.nan]) if c is None else c for c in center]
        x, y = np.array(center).T

        # insert keys
        keys = [
            dict(key, pupil_type="radius", pupil_trace=radius),
            dict(key, pupil_type="center_x", pupil_trace=x),
            dict(key, pupil_type="center_y", pupil_trace=y),
        ]
        self.insert(keys)

    def fill_pupils(self):
        from foundation.recording.trace import ScanPupil, TraceLink, TraceHomogeneous, TraceTrials

        # scan pupil traces
        ScanPupil.insert(self.proj(), skip_duplicates=True)

        # trace link
        TraceLink.fill()

        # compute trace
        key = TraceLink.ScanUnit & self
        TraceHomogeneous.populate(key, display_progress=True, reserve_jobs=True)
        TraceTrials.populate(key, display_progress=True, reserve_jobs=True)


@schema.computed
class PupilNans:
    definition = """
    -> experiment.Scan
    -> pipe_eye.FittedPupil
    -> pipe_stim.Trial
    ---
    nans                : float             # fraction of nans
    """

    @property
    def key_source(self):
        return pipe_eye.FittedPupil.proj() & experiment.Scan

    def make(self, key):
        from foundation.utils.resample import Nans

        # trace timing
        times = (experiment.Scan & key).fetch1("eye_times")
        period = np.nanmedian(np.diff(times))

        # trace value
        values = PupilTrace & dict(key, pupil_type="radius")
        values = values.fetch1("pupil_trace")

        # nan detector
        nans = Nans(times, values, period)

        # trials
        trials = pipe_stim.Trial & key
        trials, flips = trials.fetch("trial_idx", "flip_times", order_by="trial_idx", squeeze=True)

        keys = []
        for trial, flip in zip(trials, flips):

            # nans in trial
            n = nans(flip[0], flip[-1])
            _key = dict(key, trial_idx=trial, nans=n.mean())

            keys.append(_key)

        self.insert(keys)
