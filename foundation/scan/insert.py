from djutils import keys
from foundation.virtual.bridge import pipe_exp, pipe_eye, pipe_shared
from foundation.scan.experiment import Scan as ScanExperiment
from foundation.scan.pupil import PupilTrace, PupilNans
from foundation.scan.trial import TrialFilterSet, FilteredTrials
from foundation.scan.unit import UnitFilterSet, UnitSet, FilteredUnits


@keys
class Scan:
    """Scan experiment"""

    @property
    def key_list(self):
        return [
            pipe_exp.Scan,
            pipe_eye.FittedPupil,
            TrialFilterSet,
            UnitFilterSet,
        ]

    def fill(self):
        # scan timing
        ScanExperiment.populate(self.key, reserve_jobs=True, display_progress=True)

        # scan pupil
        PupilTrace.populate(self.key, reserve_jobs=True, display_progress=True)
        PupilNans.populate(self.key, reserve_jobs=True, display_progress=True)

        # scan trial set
        FilteredTrials.populate(self.key, reserve_jobs=True, display_progress=True)

        # scan unit set
        FilteredUnits.populate(self.key, reserve_jobs=True, display_progress=True)
