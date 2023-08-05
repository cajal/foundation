from djutils import keys
from foundation.virtual.bridge import pipe_exp, pipe_eye


@keys
class VisualScanExperiment:
    """Visual Scan Experiment"""

    @property
    def keys(self):
        return [
            pipe_exp.Scan,
            pipe_eye.FittedPupil,
        ]

    def fill(self):
        from foundation.scan.experiment import Scan as ScanExperiment
        from foundation.scan.pupil import PupilTrace, PupilNans

        # scan timing
        ScanExperiment.populate(self.key, reserve_jobs=True, display_progress=True)

        # scan pupil
        PupilTrace.populate(self.key, reserve_jobs=True, display_progress=True)
        PupilNans.populate(self.key, reserve_jobs=True, display_progress=True)
