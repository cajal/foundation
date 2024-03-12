from djutils import rowproperty
from foundation.schemas import fnn as schema


# ----------------------------- Modulation -----------------------------

# -- Modulation Interface --


class ModulationType:
    """Modulation Network"""

    @rowproperty
    def nn(self):
        """
        Returns
        -------
        fnn.model.modulations.Modulation
            modulation component
        """
        raise NotImplementedError()


# -- Modulation Types --


@schema.lookup
class FlatLstm(ModulationType):
    definition = """
    in_features         : int unsigned      # in features per stream
    out_features        : int unsigned      # out features per stream
    hidden_features     : int unsigned      # hidden features per stream
    init_input          : decimal(6, 4)     # initial input gate bias
    init_forget         : decimal(6, 4)     # initial forget gate bias
    dropout             : decimal(6, 6)     # dropout probability
    """

    @rowproperty
    def nn(self):
        from fnn.model.modulations import FlatLstm

        return FlatLstm(**self.fetch1("KEY"))


@schema.lookup
class SigmoidLstm(ModulationType):
    definition = """
    in_features         : int unsigned      # in features per stream
    out_features        : int unsigned      # out features per stream
    hidden_features     : int unsigned      # hidden features per stream
    init_input          : decimal(6, 4)     # initial input gate bias
    init_forget         : decimal(6, 4)     # initial forget gate bias
    dropout             : decimal(6, 6)     # dropout probability
    """

    @rowproperty
    def nn(self):
        from fnn.model.modulations import SigmoidLstm

        return SigmoidLstm(**self.fetch1("KEY"))


@schema.lookup
class MlpLstm(ModulationType):
    definition = """
    mlp_features        : int unsigned      # mlp features per stream
    mlp_layers          : int unsigned      # mlp layers
    mlp_nonlinear       : varchar(128)      # mlp nonlinearity
    lstm_features       : int unsigned      # lstm features per stream
    init_input          : decimal(6, 4)     # initial input gate bias
    init_forget         : decimal(6, 4)     # initial forget gate bias
    dropout             : decimal(6, 6)     # dropout probability
    """

    @rowproperty
    def nn(self):
        from fnn.model.modulations import MlpLstm

        return MlpLstm(**self.fetch1("KEY"))


# -- Modulation --


@schema.link
class Modulation:
    links = [FlatLstm, SigmoidLstm, MlpLstm]
    name = "modulation"
