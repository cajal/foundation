from djutils import keys
from foundation.schemas.pipeline import pipe_stim, pipe_tread, pipe_shared
from foundation.utility.resample import RateLink, OffsetLink, ResampleLink
from foundation.utility.stat import SummaryLink
from foundation.scan.experiment import Scan
from foundation.scan.pupil import PupilTrace
from foundation.scan.unit import UnitSet, FilteredUnits
from foundation.recording.scan import (
    FilteredScanTrials,
    FilteredScanPerspectives,
    FilteredScanModulations,
    FilteredScanUnits,
)
from foundation.recording.trial import (
    ScanTrial,
    TrialLink,
    TrialSet,
    TrialBounds,
    TrialVideo,
)
from foundation.recording.trace import (
    ScanUnit,
    ScanPupil,
    ScanTreadmill,
    TraceLink,
    TraceSet,
    TraceHomogeneous,
    TraceTrials,
)
from foundation.recording.stat import TraceSummary


@keys
class FillScanTrial:
    keys = [Scan]

    def fill(self):
        # trial keys
        key = pipe_stim.Trial & self.key
        ScanTrial.insert(key.proj(), skip_duplicates=True)

        # trial link
        TrialLink.fill()

        # computed trial
        key = TrialLink.ScanTrial & key
        TrialBounds.populate(key, display_progress=True, reserve_jobs=True)
        TrialVideo.populate(key, display_progress=True, reserve_jobs=True)


@keys
class FillScanTreadmill:
    keys = [Scan]

    def fill(self):
        # trace keys
        key = pipe_tread.Treadmill & self.key
        ScanTreadmill.insert(key.proj(), skip_duplicates=True)

        # trace link
        TraceLink.fill()

        # computed trace
        key = TraceLink.ScanTreadmill & key
        TraceHomogeneous.populate(key, display_progress=True, reserve_jobs=True)
        TraceTrials.populate(key, display_progress=True, reserve_jobs=True)


@keys
class FillScanPupil:
    keys = [PupilTrace]

    def fill(self):
        # trace keys
        key = PupilTrace & self.key
        ScanPupil.insert(key.proj(), skip_duplicates=True)

        # trace link
        TraceLink.fill()

        # computed trace
        key = TraceLink.ScanPupil & key
        TraceHomogeneous.populate(key, display_progress=True, reserve_jobs=True)
        TraceTrials.populate(key, display_progress=True, reserve_jobs=True)


@keys
class FillScanUnit:
    keys = [FilteredUnits, pipe_shared.SpikeMethod]

    def fill(self):
        # trace keys
        key = FilteredUnits & self.key
        key = (UnitSet & key).members * self.key
        ScanUnit.insert(key, skip_duplicates=True, ignore_extra_fields=True)

        # trace link
        TraceLink.fill()

        # computed trace
        key = TraceLink.ScanUnit & key
        TraceHomogeneous.populate(key, display_progress=True, reserve_jobs=True)
        TraceTrials.populate(key, display_progress=True, reserve_jobs=True)


@keys
class FillScanPerspectiveSummary:
    keys = [FilteredScanTrials, FilteredScanPerspectives, RateLink, OffsetLink, ResampleLink, SummaryLink]

    def fill(self):
        # keys
        trace_keys = TraceSet.Link & (FilteredScanPerspectives & self.key)
        trial_keys = TrialSet & (FilteredScanTrials & self.key)

        # trace summary statistic
        TraceSummary.populate(self.key, trial_keys, trace_keys, reserve_jobs=True, display_progress=True)


@keys
class FillScanModulationSummary:
    keys = [FilteredScanTrials, FilteredScanModulations, RateLink, OffsetLink, ResampleLink, SummaryLink]

    def fill(self):
        # keys
        trace_keys = TraceSet.Link & (FilteredScanModulations & self.key)
        trial_keys = TrialSet & (FilteredScanTrials & self.key)

        # trace summary statistic
        TraceSummary.populate(self.key, trial_keys, trace_keys, reserve_jobs=True, display_progress=True)


@keys
class FillScanUnitSummary:
    keys = [FilteredScanTrials, FilteredScanUnits, RateLink, OffsetLink, ResampleLink, SummaryLink]

    def fill(self):
        # keys
        trace_keys = TraceSet.Link & (FilteredScanUnits & self.key)
        trial_keys = TrialSet & (FilteredScanTrials & self.key)

        # trace summary statistic
        TraceSummary.populate(self.key, trial_keys, trace_keys, reserve_jobs=True, display_progress=True)
