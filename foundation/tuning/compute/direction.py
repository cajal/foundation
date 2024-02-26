from djutils import rowproperty, keys, merge
from foundation.virtual import recording, fnn, utility, stimulus, tuning
import numpy as np
import pandas as pd


# ----------------------------- DirectionResp -----------------------------

# -- DirectionResp Interface --


class DirectionResp:
    """Unit responses to direction stimuli"""

    @rowproperty
    def unit_responses(self):
        """
        Returns
        -------
        numpy.ndarray
            directions presented (degrees), sorted
        numpy.ndarray
            list of unit-wise mean responses to directions, direction x units, sorted by direction and traceset_index
        numpy.ndarray
            trace_id
        """
        raise NotImplementedError()


# -- DirectionResp Types --
    

@keys
class RecordingDirectionResp(DirectionResp):
    """Recording DirectionResp"""

    @property
    def keys(self):
        return [
            recording.TraceSet,
            recording.TrialFilterSet,
            stimulus.VideoSet,
            utility.Offset,
            utility.Impulse,
            utility.Precision,
        ]
    
    @rowproperty
    def unit_responses(self):
        from foundation.recording.trace import TraceSet

        unit_rel = (TraceSet & self.item).members
        assert len(recording.ScanUnits & unit_rel) > 0, "trace set unsupported"
        unit_rel = merge(unit_rel, recording.VisualDirectionTuning)

        directions, mean_resp, trace_ids = unit_rel.fetch("directions", "mean", "trace_id", order_by="traceset_index")

        # check if all units have the same directions
        directions = np.stack(directions, axis=1)  # direction x unit
        assert np.all(directions == directions[:, 0][:, None])
        directions = directions[:, 0]

        return directions, np.stack(mean_resp, axis=1), trace_ids


@keys
class FnnDirectionResp(DirectionResp):
    """Model DirectionResp"""

    @property
    def keys(self):
        return [
            fnn.Model,
            stimulus.VideoSet,
            utility.Burnin,
            utility.Offset,
            utility.Impulse,
            utility.Precision,
        ]
    
    @rowproperty
    def unit_responses(self):
        from foundation.fnn.data import Data
        from foundation.recording.trace import TraceSet
        direction, mean = (fnn.VisualDirectionTuning & self.item).fetch("direction", "mean", order_by="unit")
        direction = np.stack(direction, axis=1)  # direction x unit
        mean = np.stack(mean, axis=1)  # direction x unit
        assert np.all(direction == direction[:, 0][:, None]), "directions must be the same for all units"
        direction = direction[:, 0]
        # get trialset_id
        data = (Data & self.item).link.compute
        # get trial_ids
        trace_ids = (TraceSet & (data.unit_set & data.key_unit)).members.fetch('trace_id', order_by='traceset_index')
        return direction, mean, trace_ids
    

# -- gOSI --
@keys
class GOSI:
    """Global Orientation Selectivity Index"""

    @property
    def keys(self):
        return [
            tuning.DirectionResp,
        ]

    @rowproperty
    def gosi(self):
        """
        Returns
        -------
        float
            global orientation selectivity index per unit
        """
        from foundation.tuning.direction import DirectionResp
        direction, response, trace_ids = (DirectionResp & self.item).link.compute.unit_responses

        f1 = (np.exp(direction / 90 * np.pi * 1j)[:, None] * response).sum(axis=0)
        f0 = response.sum(axis=0)
        gosi = np.abs(f1) / f0

        return gosi, trace_ids


# -- gDSI --
@keys
class GDSI():

    @property
    def keys(self):
        return [
            tuning.DirectionResp,
        ]

    @rowproperty
    def gdsi(self):
        from foundation.tuning.direction import DirectionResp
        direction, response, trace_id = (DirectionResp & self.item).link.compute.unit_responses

        f1 = (np.exp(direction / 180 * np.pi * 1j)[:, None] * response).sum()
        f0 = response.sum(axis=0)
        gdsi = np.abs(f1) / f0

        return gdsi, trace_id


# -- BiVonMisesFit --
@keys
class BiVonMises():

    @property
    def keys(self):
        return [
            tuning.DirectionResp,
        ]

    @staticmethod
    def uniform(x, amp):
        return np.full_like(x, amp)

    @staticmethod
    def von_mises(x, mu, kappa, amp):
        cos_x = np.cos(x - mu)
        return amp * np.exp(kappa * cos_x - kappa)

    @staticmethod
    def bi_von_mises(x, mu, kappa, phi, amp):
        a = phi * BiVonMises.von_mises(x, mu, kappa, amp)
        b = (1 - phi) * BiVonMises.von_mises(x + np.pi, mu, kappa, amp)
        return a + b

    @rowproperty
    def fit(self):
        from foundation.tuning.direction import DirectionResp
        from foundation.utils import tqdm
        from lmfit import Model
        direction, response, trace_ids = (DirectionResp & self.item).link.compute.unit_responses

        x = direction / 180 * np.pi
        rows = []
        for (r, tr) in zip(response.T, tqdm(trace_ids)):
            mu = x[r.argmax()]

            rmin = r.min()
            rmax = r.max()
            u_amp = np.maximum(rmin, rmax / 100)
            g_amp = (rmax - u_amp) * 2

            bi_von_mises = Model(self.bi_von_mises, independent_vars=["x"], prefix="g_")
            uniform = Model(self.uniform, independent_vars=["x"], prefix="u_")
            lmodel = bi_von_mises + uniform

            params = lmodel.make_params(
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

            fit = lmodel.fit(r, params, x=x)
            row = {}
            row["success"] = int(fit.success)
            row["kappa"] = fit.params["g_kappa"].value
            row["scale"] = fit.params["g_amp"].value
            row["bias"] = fit.params["u_amp"].value
            row["mse"] = fit.chisqr / fit.ndata

            mu = fit.params["g_mu"].value
            phi = fit.params["g_phi"].value
            if phi > 0.5:
                row["mu"] = mu % (2 * np.pi)
                row["phi"] = phi
            else:
                row["mu"] = (mu + np.pi) % (2 * np.pi)
                row["phi"] = 1 - phi
            
            # compute osi from fitted model
            pref_ori = row["mu"] % np.pi
            orth_ori = (row["mu"] + np.pi / 2) % np.pi
            r_pref_ori = (
                fit.eval(x=pref_ori) +
                fit.eval(x=pref_ori + np.pi)
            ) / 2
            r_orth_ori = (
                fit.eval(x=orth_ori) +
                fit.eval(x=orth_ori + np.pi)
            ) / 2
            row["osi"] = (r_pref_ori - r_orth_ori) / (r_pref_ori + r_orth_ori)
            # compute dsi from fitted model
            r_pref_dir = fit.eval(x=row["mu"])
            r_null_dir = fit.eval(x=row["mu"] + np.pi)
            row["dsi"] = 1 - (r_null_dir / r_pref_dir)
            row["trace_id"] = tr
            rows.append(row)
        fit_rslt = pd.DataFrame(rows)
        return fit_rslt
