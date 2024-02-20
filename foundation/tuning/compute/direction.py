from djutils import rowproperty, keys, merge
from foundation.virtual import recording, fnn, utility, stimulus
import numpy as np
import pandas as pd


# ----------------------------- DirectionResp -----------------------------

# -- DirectionResp Interface --


class DirectionResp:
    """Unit responses to direction stimuli"""

    @rowproperty
    def unit_responses(self):
        """
        Returns
        -------
        numpy.ndarray
            directions presented (degrees), sorted
        numpy.ndarray
            list of unit-wise mean responses to directions, direction x units, sorted by direction and traceset_index
        numpy.ndarray
            trace_id
        """
        raise NotImplementedError()


# -- DirectionResp Types --
    

@keys
class RecordingDirectionResp(DirectionResp):
    """Recording DirectionResp"""

    @property
    def keys(self):
        return [
            recording.TraceSet,
            recording.TrialFilterSet,
            stimulus.VideoSet,
            utility.Offset,
            utility.Impulse,
            utility.Precision,
        ]
    
    @rowproperty
    def unit_responses(self):
        from foundation.recording.trace import TraceSet

        unit_rel = (TraceSet & self.item).members
        assert len(recording.ScanUnits & unit_rel) > 0, "trace set unsupported"
        units_rel = merge(unit_rel, recording.VisualDirectionTuning)

        directions, mean_resp, trace_ids = unit_rel.fetch("directions", "mean", "trace_id", order_by="traceset_index")

        # check if all units have the same directions
        directions = np.stack(directions)  # direction x unit
        assert np.all(directions == directions[:, 0][:, None])
        directions = directions[:, 0]

        return directions, np.stack(mean_resp, axis=1), trace_ids


@keys
class FnnDirectionResp(DirectionResp):
    """Model DirectionResp"""

    @property
    def keys(self):
        return [
            fnn.Model,
            stimulus.VideoSet,
            utility.Burnin,
            utility.Offset,
            utility.Impulse,
            utility.Precision,
        ]
    
    @rowproperty
    def unit_responses(self):
        from foundation.fnn.data import Data
        from foundation.recording.trace import TraceSet
        direction, mean = (fnn.VisualDirectionTuning & self.item).fetch("direction", "mean")
        # get trialset_id
        data = (Data & self.item).link.compute
        # get trial_ids
        trace_ids = (TraceSet & (data.unit_set & data.key_unit)).members.fetch('trace_id', order_by='traceset_index')
        return direction, mean, trace_ids
    

# # -- OSI --
# @keys
# class OSI:
#     """Orientation Selectivity Index"""

#     @property
#     def keys(self):
#         return [
#             DirectionResp,
#             recording.Trace,
#         ]
    
#     @rowproperty
#     def osi(self):
#         """
#         Returns
#         -------
#         float
#             orientation selectivity index
#         float
#             global orientation selectivity index
#         """
#         raise NotImplementedError()