from djutils import keys
from foundation.virtual import recording
from foundation.dataset.dtype import ScanUnit, ScanPerspective, ScanModulation, Dtype
from foundation.dataset.recording import Scan as ScanRecording, Recording, RecordingTrials, RecordingTraces


@keys
class Scan:
    """Scan dataset"""

    @property
    def key_list(self):
        return [
            recording.ScanTrials,
            recording.ScanPerspectives,
            recording.ScanModulations,
            recording.ScanUnits,
        ]

    def fill(self):
        # scan dtypes
        for dtype in [ScanUnit, ScanPerspective, ScanModulation]:
            dtype.insert(self.key, skip_duplicates=True, ignore_extra_fields=True)

        # dtype link
        Dtype.fill()

        # scan recording dataset
        ScanRecording.insert(self.key, skip_duplicates=True, ignore_extra_fields=True)

        # recording link
        Recording.fill()

        # computed recording
        key = Recording.Scan & self.key
        RecordingTrials.populate(key, reserve_jobs=True, display_progress=True)
        RecordingTraces.populate(key, reserve_jobs=True, display_progress=True)
