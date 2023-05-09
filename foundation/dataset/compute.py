import pandas as pd
from djutils import keys, merge
from foundation.utility.resample import Rate, Offset, Resample
from foundation.recording.trial import TrialSet
from foundation.recording.trace import TraceSet
from foundation.recording.cache import ResampledVideo, ResampledTraces
from foundation.dataset.recording import RecordingTrials, RecordingTraces


@keys
class LoadRecordingTrials:
    """Load resampled recording trials"""

    @property
    def key_list(self):
        return [
            RecordingTrials,
            Rate,
        ]


@keys
class LoadRecordingTraces:
    """Load resampled recording traces"""

    @property
    def key_list(self):
        return [
            RecordingTrials,
            RecordingTraces,
            Rate,
            Offset,
            Resample,
        ]
