from foundation.recording.insert import (
    Scan,
    ScanBehaviorSummary,
    ScanUnitSummary,
    ScanVideoCache,
    ScanBehaviorCache,
    ScanUnitCache,
)
from foundation.virtual import utility

# foundation cohort
cohort = [
    {"animal_id": 24620, "session": 9, "scan_idx": 13},
    {"animal_id": 25133, "session": 12, "scan_idx": 14},
    {"animal_id": 25312, "session": 2, "scan_idx": 24},
    {"animal_id": 25404, "session": 4, "scan_idx": 20},
    {"animal_id": 25505, "session": 3, "scan_idx": 11},
    {"animal_id": 25702, "session": 5, "scan_idx": 16},
    {"animal_id": 25830, "session": 3, "scan_idx": 9},
    {"animal_id": 25833, "session": 3, "scan_idx": 13},
]

(Scan & cohort & {"spike_method": 6}).fill()

(ScanBehaviorSummary & cohort & utility.Resample.LowpassHamming).fill()

(ScanUnitSummary & cohort & utility.Resample.Hamming & utility.Summary.Mean).fill()

(ScanVideoCache & cohort).fill()

(ScanBehaviorCache & cohort & utility.Resample.LowpassHamming).fill()

(ScanUnitCache & cohort & utility.Resample.Hamming).fill()
