from djutils import rowproperty
from foundation.fnn.train import Train
from foundation.fnn.transfer import TransferList
from foundation.schemas import fnn as schema


# ----------------------------- Instance -----------------------------

# -- Instance Interface --


class InstanceType:
    """Instance"""

    @rowproperty
    def compute(self):
        """
        Returns
        -------
        foundation.fnn.compute_instance.InstanceType (row)
            compute instance
        """
        raise NotImplementedError()


# -- Instance Types --


@schema.lookup
class Individual:
    definition = """
    -> TransferList
    -> Train
    parallel        : int unsigned      # parallel group size
    cycle           : int unsigned      # training cycle
    seed            : int unsigned      # seed for initialization
    """


# -- Instance --


@schema.link
class Instance:
    links = [Individual]
    name = "instance"
    comment = "fnn instance"
