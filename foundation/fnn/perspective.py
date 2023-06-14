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


# ----------------------------- Pixel Intensity -----------------------------

# -- Pixel Intensity Base --


class _Pixel:
    """Pixel Intensity"""

    @rowproperty
    def nn(self):
        """
        Returns
        -------
        fnn.model.pixels.Pixel
            pixel intensity
        """
        raise NotImplementedError()


# -- Luminance Types --


@schema.method
class Raw(_Pixel):
    name = "raw"
    comment = "raw pixel intensity"

    @rowproperty
    def nn(self):
        from fnn.model.pixels import Raw

        return Raw()


# -- Pixel Intensity --


@schema.link
class Pixel:
    links = [Raw]
    name = "pixel"


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

    @rowproperty
    def dropout(self):
        """
        Returns
        -------
        float
            dropout probability
        """
        return 0


# -- Perspective Types --


@schema.lookup
class MonitorRetina(_Perspective):
    definition = """
    -> Monitor
    -> Pixel.proj(monitor_pixel_id="pixel_id")
    -> Retina
    -> Pixel.proj(retina_pixel_id="pixel_id")
    height          : int unsigned  # retina height
    width           : int unsigned  # retina width
    features        : varchar(128)  # mlp features (csv)
    nonlinear       : varchar(128)  # nonlinearity
    """

    @rowproperty
    def nn(self):
        from fnn.model.perspectives import MonitorRetina

        monitor_key = self.proj(pixel_id="monitor_pixel_id")
        retina_key = self.proj(pixel_id="retina_pixel_id")
        height, width, features, nonlinear = self.fetch1("height", "width", "features", "nonlinear")

        return MonitorRetina(
            monitor=(Monitor & self).link.nn,
            monitor_pixel=(Pixel & monitor_key).link.nn,
            retina=(Retina & self).link.nn,
            retina_pixel=(Pixel & retina_key).link.nn,
            height=height,
            width=width,
            features=features.split(","),
            nonlinear=nonlinear,
        )


# -- Perspective --


@schema.link
class Perspective:
    links = [MonitorRetina]
    name = "perspective"
