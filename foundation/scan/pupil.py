import numpy as np
import datajoint as dj
from foundation.utility import resample
from foundation.scan import timing
from foundation.schemas.pipeline import pipe_eye, pipe_stim
from foundation.schemas import scan as schema


@schema
class Pupil(dj.Computed):
    definition = """
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


@schema
class PupilNans(dj.Computed):
    definition = """
    -> pipe_eye.FittedPupil
    -> pipe_stim.Trial
    -> resample.RateLink
    -> resample.OffsetLink
    ---
    nans        : int unsigned      # number of nans
    """

    @property
    def key_source(self):
        keys = pipe_eye.FittedPupil.proj() * resample.RateLink.proj() * resample.OffsetLink.proj()
        return keys & Pupil

    def make(self, key):
        from foundation.utils.trace import Nans

        # pupil trace times and values
        times = (timing.Times & key).fetch1("eye_times")
        values = (Pupil & dict(key, pupil_type="radius")).fetch1("pupil_trace")

        # sampling rate
        rate_link = (resample.RateLink & key).link
        period = rate_link.period

        # nan detector
        nans = Nans(times, values, period)

        # trials
        trials = pipe_stim.Trial & key
        trials, flips = trials.fetch("trial_idx", "flip_times", order_by="trial_idx", squeeze=True)

        keys = []
        for trial, flip in zip(trials, flips):

            # number of nans in trial
            samples = rate_link.samples(flip[-1] - flip[0])
            n = nans(flip[0], samples).sum()

            _key = dict(key, trial_idx=trial, nans=n)
            keys.append(_key)

        self.insert(keys)
