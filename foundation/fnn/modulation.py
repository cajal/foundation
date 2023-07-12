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
    lstm_features   : int unsigned      # lstm features per stream
    out_features    : int unsigned      # out features per stream
    init_input      : decimal(6, 4)     # initial input gate bias
    init_forget     : decimal(6, 4)     # initial forget gate bias
    dropout         : decimal(6, 6)     # dropout probability
    """

    @rowproperty
    def nn(self):
        from fnn.model.modulations import FlatLstm

        return FlatLstm(**self.fetch1("KEY"))


# -- Modulation --


@schema.link
class Modulation:
    links = [FlatLstm]
    name = "modulation"
