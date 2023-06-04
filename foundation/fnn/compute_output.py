from djutils import keys, rowproperty
from foundation.virtual import stimulus, recording, fnn


# ----------------------------- Visual Output -----------------------------


@keys
class Visual:
    """Visual Model Network Output"""

    @rowproperty
    def responses(self):
        pass
