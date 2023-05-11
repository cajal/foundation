from djutils import rowproperty
from foundation.schemas import fnn as schema


# -------------- Monitor --------------

# -- Monitor Base --


class _Monitor:
    """Monitor"""

    @rowproperty
    def nn(self):
        """
        Returns
        -------
        fnn.model.monitors.Monitor
            fnn monitor
        """
        raise NotImplementedError()


# -- Monitor Types --


@schema.lookup
class Plane(_Monitor):
    definition = """
    init_center_x       : decimal(6, 4)     # initial center x
    init_center_y       : decimal(6, 4)     # initial center y
    init_center_z       : decimal(6, 4)     # initial center z
    init_center_std     : decimal(6, 4)     # initial center stddev
    init_angle_x        : decimal(6, 4)     # initial angle x
    init_angle_y        : decimal(6, 4)     # initial angle y
    init_angle_z        : decimal(6, 4)     # initial angle z
    init_angle_std      : decimal(6, 4)     # initial angle stddev
    """

    @rowproperty
    def nn(self):
        from fnn.model.monitors import Plane

        return Plane(**{k: float(v) for k, v in self.fetch1().items()})


# -- Monitor --


@schema.link
class Monitor:
    links = [Plane]
    name = "monitor"
    comment = "monitor"


# -------------- Retina --------------

# -- Retina Base --


class _Retina:
    """Retina"""

    @rowproperty
    def nn(self):
        """
        Returns
        -------
        fnn.model.retinas.Retina
            fnn retina
        """
        raise NotImplementedError()


# -- Retina Types --


@schema.lookup
class Angular(_Retina):
    definition = """
    degrees         : decimal(6, 3)     # maximum visual degrees
    """

    @rowproperty
    def nn(self):
        from fnn.model.retinas import Angular

        return Angular(degrees=float(self.fetch1("degrees")))


# -- Retina --


@schema.link
class Retina:
    links = [Angular]
    name = "retina"
    comment = "retina"


# -------------- Perspective --------------

# -- Perspective Base --


class _Perspective:
    """Perspective"""

    @rowproperty
    def nn(self):
        """
        Returns
        -------
        fnn.model.perspectives.Perspective
            fnn perspective
        """
        raise NotImplementedError()


# -- Perspective Types --


@schema.lookup
class MonitorRetina(_Perspective):
    definition = """
    -> Monitor
    -> Retina
    features        : varchar(128)  # MLP features (csv)
    nonlinear       : varchar(128)  # nonlinearity
    """

    @rowproperty
    def nn(self):
        from fnn.model.perspectives import MonitorRetina

        monitor = (Monitor & self).link.nn
        retina = (Retina & self).link.nn

        features, nonlinear = self.fetch1("features", "nonlinear")
        features = list(map(int, features.split(",")))

        return MonitorRetina(monitor=monitor, retina=retina, features=features, nonlinear=nonlinear)


# -- Perspective --


@schema.link
class Perspective:
    links = [MonitorRetina]
    name = "perspective"
    comment = "perspective"
