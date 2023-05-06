from foundation.recording.insert import *

key = ScanUnitCache & "animal_id=27203 and session=4 and scan_idx=7" & utility.ResampleLink.Hamming
key.fill(10)
