from djutils import merge, rowproperty, rowmethod
from foundation.virtual import recording, fnn, stimulus, utility
from foundation.schemas import tuning as schema


# ---------------------------- DirectionResp ----------------------------

# -- DirectionResp Interface --


class DirectionResp:
    """Unit responses to direction stimuli"""

    @rowproperty
    def compute(self):
        """
        Returns
        -------
        foundation.tuning.compute.direction (row)
            compute unit responses to direction stimuli
        """
        raise NotImplementedError()


# -- DirectionResp Types --


@schema.lookup
class RecordingDirectionResp(DirectionResp):
    definition = """
    -> recording.TraceSet
    -> recording.TrialFilterSet
    -> stimulus.VideoSet
    -> utility.Offset
    -> utility.Impulse
    -> utility.Precision
    """

    @rowproperty
    def compute(self):
        from foundation.tuning.compute.direction import RecordingDirectionResp

        return RecordingDirectionResp & self


@schema.lookup
class FnnDirectionResp(DirectionResp):
    definition = """
    -> fnn.Model
    -> stimulus.VideoSet
    -> utility.Burnin
    -> utility.Offset
    -> utility.Impulse
    -> utility.Precision
    """

    @rowproperty
    def compute(self):
        from foundation.tuning.compute.direction import FnnDirectionResp

        return FnnDirectionResp & self


# -- DirectionResp --


@schema.link
class DirectionResp:
    links = [RecordingDirectionResp, FnnDirectionResp]
    name = "direction_resp"
    comment = "unit responses to direction stimuli"


# -- gOSI --


@schema.computed
class GOSI:
    definition = """
    -> DirectionResp
    -> recording.Trace
    ---
    gosi = NULL     : float     # global orientation selectivity index
    """

    @property
    def key_source(self):
        from foundation.tuning.compute.direction import GOSI

        return GOSI.key_source

    def make(self, key):
        from foundation.tuning.compute.direction import GOSI

        gosi, trace_ids = (GOSI & key).gosi
        self.insert(
            [{**key, "trace_id": trace, "gosi": g} for trace, g in zip(trace_ids, gosi)]
        )


# -- gDSI --


@schema.computed
class GDSI:
    definition = """
    -> DirectionResp
    -> recording.Trace
    ---
    gdsi = NULL     : float     # global direction selectivity index
    """

    @property
    def key_source(self):
        from foundation.tuning.compute.direction import GDSI

        return GDSI.key_source

    def make(self, key):
        from foundation.tuning.compute.direction import GDSI

        gdsi, trace_ids = (GDSI & key).gdsi
        self.insert(
            [{**key, "trace_id": trace, "gdsi": g} for trace, g in zip(trace_ids, gdsi)]
        )


# -- BiVonMisesFit --
@schema.computed
class BiVonMises():
    definition = """
    -> DirectionResp
    -> recording.Trace
    ---
    success     : bool      # success of least squares optimization
    mu          : float     # center of the first von Mises distribution
    phi         : float     # weight of the first von Mises distribution
    kappa       : float     # dispersion of both von Mises distributions
    scale       : float     # von Mises amplitude
    bias        : float     # uniform amplitude
    mse         : float     # mean squared error
    osi         : float     # orientation selectivity index
    dsi         : float     # direction selectivity index
    """

    @property
    def key_source(self):
        from foundation.tuning.compute.direction import BiVonMises
        return BiVonMises.key_source

    
    def make(self, key):
        from foundation.tuning.compute.direction import BiVonMises
        fit = (BiVonMises & key).fit
        for k, v in key.items():
            fit[k] = v
        self.insert(fit)