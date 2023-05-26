from djutils import rowproperty
from foundation.schemas import fnn as schema


# ----------------------------- Monitor -----------------------------

# -- Monitor Base --


class _Monitor:
    """Monitor"""

    @rowproperty
    def nn(self):
        """
        Returns
        -------
        fnn.model.monitors.Monitor
            monitor component
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


# ----------------------------- Luminance -----------------------------

# -- Luminance Base --


class _Luminance:
    """Luminance"""

    @rowproperty
    def nn(self):
        """
        Returns
        -------
        fnn.model.luminances.Luminance
            luminance component
        """
        raise NotImplementedError()


# -- Luminance Types --


@schema.lookup
class Power(_Luminance):
    definition = """
    power           : decimal(6, 4)     # luminance power
    scale           : decimal(6, 4)     # luminance scale
    offset          : decimal(6, 4)     # luminance offset
    """

    @rowproperty
    def nn(self):
        from fnn.model.luminances import Power

        p, s, o = map(float, self.fetch1("power", "scale", "offset"))

        return Power(power=p, scale=s, offset=o)


# -- Luminance --


@schema.link
class Luminance:
    links = [Power]
    name = "luminance"


# ----------------------------- Retina -----------------------------

# -- Retina Base --


class _Retina:
    """Retina"""

    @rowproperty
    def nn(self):
        """
        Returns
        -------
        fnn.model.retinas.Retina
            retina component
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


# ----------------------------- Perspective -----------------------------

# -- Perspective Base --


class _Perspective:
    """Perspective"""

    @rowproperty
    def nn(self):
        """
        Returns
        -------
        fnn.model.perspectives.Perspective
            perspective component
        """
        raise NotImplementedError()


# -- Perspective Types --


@schema.lookup
class MonitorRetina(_Perspective):
    definition = """
    -> Monitor
    -> Luminance
    -> Retina
    height          : int unsigned  # retina height
    width           : int unsigned  # retina width
    features        : varchar(128)  # mlp features (csv)
    nonlinear       : varchar(128)  # nonlinearity
    """

    @rowproperty
    def nn(self):
        from fnn.model.perspectives import MonitorRetina

        m = (Monitor & self).link.nn
        l = (Luminance & self).link.nn
        r = (Retina & self).link.nn
        h, w, f, n = self.fetch1("height", "width", "features", "nonlinear")
        f = list(map(int, f.split(",")))

        return MonitorRetina(monitor=m, luminance=l, retina=r, height=h, width=w, features=f, nonlinear=n)


# -- Perspective --


@schema.link
class Perspective:
    links = [MonitorRetina]
    name = "perspective"
