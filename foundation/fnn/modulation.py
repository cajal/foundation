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
class Lstm(_Modulation):
    definition = """
    features        : int unsigned      # lstm features
    dropout         : decimal(6, 6)     # dropout probability
    """

    @rowproperty
    def nn(self):
        from fnn.model.modulations import Lstm

        return Lstm(**self.fetch1("KEY"))


# -- Modulation --


@schema.link
class Modulation:
    links = [Lstm]
    name = "modulation"
