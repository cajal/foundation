from djutils import rowproperty
from foundation.schemas import utility as schema


# ---------------------------- Precision ----------------------------

# -- Precision Base --


class PrecisionType:
    """Numeric Precision"""

    @rowproperty
    def string(self):
        """
        Returns
        -------
        Callable[[float], str]
            float -> str
        """
        raise NotImplementedError()


# -- Precision Types --


@schema.lookup
class Digits(PrecisionType):
    definition = """
    digits      : int unsigned  # decimal digits
    """

    @rowproperty
    def string(self):
        digits = self.fetch1("digits")

        return lambda x: f"{x:.{digits}f}"


# -- Precision --


@schema.link
class Precision:
    links = [Digits]
    name = "precision"
    comment = "numeric precision"
