from djutils import rowmethod
from foundation.schemas import utility as schema


# ---------------------------- Impulse ----------------------------

# -- Impulse Interface --


class ImpulseType:
    """Impulse Method"""

    @rowmethod
    def impulse(self, times, values, target_offset):
        """
        Parameters
        -------
        times : 1D array
            trace times, monotonically increasing
        values : ND array
            trace values, first dim same length as times
        target_offset : float
            target offset

        Returns
        -------
        foundation.utils.impulse.Impulse
            callable, computes impulse
        """
        raise NotImplementedError()


# -- Impulse Types --


@schema.method
class Box(ImpulseType):
    name = "box"
    comment = "box impulse"

    @rowmethod
    def impulse(self, times, values, target_offset):
        from foundation.utils.impulse import Box

        return Box(times=times, values=values, target_offset=target_offset)


# -- Impulse --


@schema.link
class Impulse:
    links = [Box]
    name = "impulse"
    comment = "impulse method"
