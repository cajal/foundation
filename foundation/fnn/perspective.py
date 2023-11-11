from djutils import rowproperty
from foundation.schemas import fnn as schema


# ----------------------------- Monitor -----------------------------

# -- Monitor Interface --


class MonitorType:
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
class Plane(MonitorType):
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


# ----------------------------- Pixel -----------------------------

# -- Pixel Interface --


class PixelType:
    """Pixel"""

    @rowproperty
    def nn(self):
        """
        Returns
        -------
        fnn.model.pixels.Pixel
            pixel intensity
        """
        raise NotImplementedError()


# -- Pixel Types --


@schema.lookup
class StaticPower(PixelType):
    definition = """
    power               : decimal(6, 4)     # pixel power
    scale               : decimal(6, 4)     # pixel scale
    offset              : decimal(6, 4)     # pixel offset
    """

    @rowproperty
    def nn(self):
        from fnn.model.pixels import StaticPower

        return StaticPower(**self.fetch1())


@schema.lookup
class SigmoidPower(PixelType):
    definition = """
    max_power           : decimal(6, 4)     # maximum pixel power
    init_scale          : decimal(6, 4)     # initial pixel scale
    init_offset         : decimal(6, 4)     # initial pixel offset
    """

    @rowproperty
    def nn(self):
        from fnn.model.pixels import SigmoidPower

        return SigmoidPower(**self.fetch1())


# -- Pixel --


@schema.link
class Pixel:
    links = [StaticPower, SigmoidPower]
    name = "pixel"


# ----------------------------- Retina -----------------------------

# -- Retina Interface --


class RetinaType:
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
class Angular(RetinaType):
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

# -- Perspective Interface --


class PerspectiveType:
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
class MonitorRetina(PerspectiveType):
    definition = """
    -> Monitor
    -> Pixel.proj(monitor_pixel_id="pixel_id")
    -> Retina
    -> Pixel.proj(retina_pixel_id="pixel_id")
    height          : int unsigned      # retina height
    width           : int unsigned      # retina width
    features        : varchar(128)      # mlp features (csv)
    nonlinear       : varchar(128)      # nonlinearity
    dropout         : decimal(6, 6)     # dropout probability
    """

    @rowproperty
    def nn(self):
        from fnn.model.perspectives import MonitorRetina

        monitor_key = self.proj(pixel_id="monitor_pixel_id")
        retina_key = self.proj(pixel_id="retina_pixel_id")
        height, width, features, nonlinear, dropout = self.fetch1(
            "height", "width", "features", "nonlinear", "dropout"
        )

        return MonitorRetina(
            monitor=(Monitor & self).link.nn,
            monitor_pixel=(Pixel & monitor_key).link.nn,
            retina=(Retina & self).link.nn,
            retina_pixel=(Pixel & retina_key).link.nn,
            height=height,
            width=width,
            features=features.split(","),
            nonlinear=nonlinear,
            dropout=dropout,
        )


@schema.lookup
class MlpMonitorRetina(PerspectiveType):
    definition = """
    -> Monitor
    -> Pixel.proj(monitor_pixel_id="pixel_id")
    -> Retina
    -> Pixel.proj(retina_pixel_id="pixel_id")
    mlp_features    : int unsigned      # mlp features
    mlp_layers      : int unsigned      # mlp layers
    mlp_nonlinear   : varchar(128)      # mlp nonlinearity
    height          : int unsigned      # retina height
    width           : int unsigned      # retina width
    """

    @rowproperty
    def nn(self):
        from fnn.model.perspectives import MlpMonitorRetina

        (
            mon_id,
            mon_pix_id,
            ret_id,
            ret_pix_id,
            mlp_features,
            mlp_layers,
            mlp_nonlinear,
            height,
            width,
        ) = self.fetch1(
            "monitor_id",
            "monitor_pixel_id",
            "retina_id",
            "retina_pixel_id",
            "mlp_features",
            "mlp_layers",
            "mlp_nonlinear",
            "height",
            "width",
        )

        return MlpMonitorRetina(
            monitor=(Monitor & {"monitor_id": mon_id}).link.nn,
            monitor_pixel=(Pixel & {"pixel_id": mon_pix_id}).link.nn,
            retina=(Retina & {"retina_id": ret_id}).link.nn,
            retina_pixel=(Pixel & {"pixel_id": ret_pix_id}).link.nn,
            mlp_features=mlp_features,
            mlp_layers=mlp_layers,
            mlp_nonlinear=mlp_nonlinear,
            height=height,
            width=width,
        )


# -- Perspective --


@schema.link
class Perspective:
    links = [MonitorRetina, MlpMonitorRetina]
    name = "perspective"
