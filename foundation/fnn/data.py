from datajoint import U
from djutils import merge, rowproperty
from foundation.virtual import utility, stimulus, recording
from foundation.schemas import fnn as schema


# ---------------------------- Spec ----------------------------

# -- Spec Types --


@schema.lookup
class VideoSpec:
    definition = """
    -> utility.Resize
    -> utility.Resolution
    """


@schema.lookup
class TraceSpec:
    definition = """
    -> utility.Standardize
    -> utility.Resample
    -> utility.Offset
    """


# -- Spec --


@schema.link
class Spec:
    links = [VideoSpec, TraceSpec]
    name = "spec"
    comment = "data specification"


# ---------------------------- Data ----------------------------


class _Data:
    """Data"""

    @rowproperty
    def dataset(self):
        """
        Returns
        -------
        fnn.data.Dataset
            network dataset
        """
        raise NotImplementedError()

    @rowproperty
    def sizes(self):
        """
        Returns
        -------
        dict[str, int]
            network sizes
        """
        raise NotImplementedError()

    # @rowproperty
    # def visual_inputs(self):
    #     """
    #     Returns
    #     -------
    #     djutils.derived.Keys
    #         key_list -- [foundation.stimulus.Video, ...]
    #         rowmethod -- [trials, stimuli, perspectives, modulations, ...]
    #     """
    #     raise NotImplementedError()

    # @rowproperty
    # def response_timing(self):
    #     """
    #     Returns
    #     -------
    #     float
    #         response period (seconds)
    #     float
    #         response offset (seconds)
    #     """
    #     raise NotImplementedError()


# -- Data Set Types --


@schema.lookup
class VisualScan(_Data):
    definition = """
    -> recording.ScanVisualPerspectives
    -> recording.ScanVisualModulations
    -> recording.ScanUnits
    -> recording.ScanTrials
    -> Spec.VideoSpec.proj(stimuli_id="spec_id")
    -> Spec.TraceSpec.proj(perspectives_id="spec_id")
    -> Spec.TraceSpec.proj(modulations_id="spec_id")
    -> Spec.TraceSpec.proj(units_id="spec_id")
    -> utility.Rate
    """

    @rowproperty
    def sizes(self):
        from foundation.fnn.compute_data import VisualScan

        sizes = dict()
        key = VisualScan & self

        stimuli = merge(key.trials, recording.TrialVideo, stimulus.VideoInfo)
        sizes["stimuli"] = (U("channels") & stimuli).fetch1("channels")

        for attr in ["perspectives", "modulations", "units"]:
            _key = recording.TraceSet & getattr(key, f"{attr}_key")
            sizes[attr] = _key.fetch1("members")

        return sizes

    @rowproperty
    def dataset(self):
        from foundation.fnn.compute_dataset import VisualScan

        return (VisualScan & self).dataset

    # @rowproperty
    # def visual_inputs(self):
    #     from foundation.fnn.compute_data import VisualScanInputs

    #     return VisualScanInputs & self

    # @rowproperty
    # def response_timing(self):
    #     from foundation.utility.resample import Rate, Offset

    #     key = self.proj(spec_id="units_id") * Spec.TraceSpec
    #     period = (Rate & key).link.period
    #     offset = (Offset & key).link.offset

    #     return period, offset


# -- Data Set Types --


@schema.link
class Data:
    links = [VisualScan]
    name = "data"
    comment = "fnn data"
