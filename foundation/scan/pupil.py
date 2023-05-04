import numpy as np
from foundation.scan.experiment import Scan
from foundation.schemas.pipeline import pipe_eye, pipe_stim
from foundation.schemas import scan as schema


@schema.computed
class PupilTrace:
    definition = """
    -> Scan
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


@schema.computed
class PupilNans:
    definition = """
    -> Scan
    -> pipe_eye.FittedPupil
    -> pipe_stim.Trial
    ---
    nans                : float             # fraction of nans
    """

    @property
    def key_source(self):
        return pipe_eye.FittedPupil.proj() & Scan

    def make(self, key):
        from foundation.utils.resample import Nans

        # trace timing
        times = (Scan & key).fetch1("eye_times")
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
