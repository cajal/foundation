import numpy as np
import pandas as pd
from djutils import rowproperty, keys

from foundation.virtual import recording, fnn, tuning


# ----------------------------- Direction Tuning -----------------------------

# -- Direction Tuning Interface --


class DirectionType:
    """Direction Response"""

    @rowproperty
    def tuning(self):
        """
        Returns
        -------
        1D array
            presented directions (degrees)
        1D array
            response to directions
        """
        raise NotImplementedError()


# -- Direction Tuning Types --


@keys
class RecordingVisualDirection(DirectionType):
    """Recording Visual Direction Tuning"""

    @property
    def keys(self):
        return [
            recording.VisualDirectionTuning,
        ]

    @rowproperty
    def tuning(self):
        return (recording.VisualDirectionTuning & self.item).fetch1("direction", "response")


@keys
class FnnVisualDirection(DirectionType):
    """Fnn Visual Direction Tuning"""

    @property
    def keys(self):
        return [
            fnn.VisualDirectionTuning,
        ]

    @rowproperty
    def tuning(self):
        return (fnn.VisualDirectionTuning & self.item).fetch1("direction", "response")


# ----------------------------- Direction Fit -----------------------------


def uniform(x, amp):
    return np.full_like(x, amp)


def von_mises(x, mu, kappa, amp):
    cos_x = np.cos(x - mu)
    return amp * np.exp(kappa * cos_x - kappa)


def bi_von_mises(x, mu, kappa, phi, amp):
    a = phi * von_mises(x, mu, kappa, amp)
    b = (1 - phi) * von_mises(x + np.pi, mu, kappa, amp)
    return a + b


@keys
class GlobalOSI:
    """Global Orientation Selectivity Index"""

    @property
    def keys(self):
        return [
            tuning.Direction,
        ]

    @rowproperty
    def global_osi(self):
        from foundation.tuning.direction import Direction

        direction, response = (Direction & self.item).link.compute.tuning

        f1 = (np.exp(direction / 90 * np.pi * 1j) * response).sum()
        f0 = response.sum()

        return np.abs(f1) / f0


@keys
class GlobalDSI:
    """Global Direction Selectivity Index"""

    @property
    def keys(self):
        return [
            tuning.Direction,
        ]

    @rowproperty
    def global_osi(self):
        from foundation.tuning.direction import Direction

        direction, response = (Direction & self.item).link.compute.tuning

        f1 = (np.exp(direction / 180 * np.pi * 1j) * response).sum()
        f0 = response.sum()

        return np.abs(f1) / f0


@keys
class BiVonMises:
    """Bi Von Mises"""

    @property
    def keys(self):
        return [
            tuning.Direction,
        ]

    @rowproperty
    def bi_von_mises(self):
        from foundation.tuning.direction import Direction
        from lmfit import Model

        direction, response = (Direction & self.item).link.compute.tuning

        x = direction / 180 * np.pi
        mu = x[response.argmax()]
        rmin = response.min()
        rmax = response.max()
        u_amp = np.maximum(rmin, rmax / 100)
        g_amp = (rmax - u_amp) * 2

        b = Model(bi_von_mises, independent_vars=["x"], prefix="g_")
        u = Model(uniform, independent_vars=["x"], prefix="u_")
        model = b + u

        params = model.make_params(
            g_mu=mu,
            g_kappa=1,
            g_phi=0.5,
            g_amp=g_amp,
            u_amp=u_amp,
        )
        params["g_phi"].set(min=0, max=1)
        params["g_kappa"].set(min=0)
        params["g_amp"].set(min=0)
        params["u_amp"].set(min=0)

        fit = model.fit(response, params, x=x)

        params = {
            "success": fit.success,
            "kappa": fit.params["g_kappa"].value,
            "scale": fit.params["g_amp"].value,
            "bias": fit.params["u_amp"].value,
            "mse": fit.chisqr / fit.ndata,
        }

        mu = fit.params["g_mu"].value
        phi = fit.params["g_phi"].value

        if phi > 0.5:
            params["mu"] = mu % (2 * np.pi)
            params["phi"] = phi
        else:
            params["mu"] = (mu + np.pi) % (2 * np.pi)
            params["phi"] = 1 - phi

        return params
