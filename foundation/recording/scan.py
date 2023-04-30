import numpy as np
import datajoint as dj
from foundation.recording import trial, trace
from foundation.schemas.pipeline import pipe_exp
from foundation.schemas import recording as schema


@schema
class ScanTrialSet(dj.Computed):
    definition = """
    -> pipe_exp.Scan
    ---
    -> trial.TrialSet
    """
