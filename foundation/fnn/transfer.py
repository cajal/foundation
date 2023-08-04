from djutils import rowmethod
from foundation.schemas import fnn as schema


# ----------------------------- Transfer -----------------------------

# -- Transfer Interface --

# -- Transfer Types --

# -- Transfer --


@schema.link
class Transfer:
    links = []
    name = "transfer"
    comment = "fnn transfer"


@schema.linklist
class TransferList:
    link = Transfer
    name = "transferlist"
    comment = "fnn transfer list"
