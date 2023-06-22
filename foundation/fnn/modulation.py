from djutils import rowproperty
from foundation.schemas import fnn as schema


# ----------------------------- Modulation -----------------------------

# -- Modulation Base --


class _Modulation:
    """Modulation"""

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
class LnLstm(_Modulation):
    definition = """
    features        : int unsigned      # feature size
    nonlinear       : varchar(128)      # nonlinearity
    dropout         : decimal(6, 6)     # dropout probability
    """

    @rowproperty
    def nn(self):
        from fnn.model.modulations import LnLstm

        return LnLstm(**self.fetch1("KEY"))


# -- Modulation --


@schema.link
class Modulation:
    links = [LnLstm]
    name = "modulation"
