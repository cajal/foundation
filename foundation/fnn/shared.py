from djutils import rowproperty
from foundation.schemas import fnn as schema


# ----------------------------- Bound -----------------------------

# -- Bound Interface --


class BoundType:
    """Bound"""

    @rowproperty
    def nn(self):
        """
        Returns
        -------
        fnn.model.bounds.Bound
            bound component
        """
        raise NotImplementedError()


# -- Bound Types --


@schema.method
class Sigmoid(BoundType):
    name = "sigmoid"
    comment = "sigmoid bound"

    @rowproperty
    def nn(self):
        from fnn.model.bounds import Sigmoid

        return Sigmoid()


@schema.method
class Tanh(BoundType):
    name = "tanh"
    comment = "tanh bound"

    @rowproperty
    def nn(self):
        from fnn.model.bounds import Tanh

        return Tanh()


# -- Bound --


@schema.link
class Bound:
    links = [Sigmoid, Tanh]
    name = "bound"
