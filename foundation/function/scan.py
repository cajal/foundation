import numpy as np
import pandas as pd
from djutils import merge
from foundation.virtual import scan, recording, fnn
from foundation.virtual.bridge import pipe_fuse
from foundation.function.response import TrialResponse, FnnTrialResponse, Response, ResponseSet
from foundation.schemas import function as schema


# @schema.computed
# class FnnVisualScan:
#     definition = """
#     -> fnn.ModelNetwork
#     ---
#     -> fnn.VisualScan
#     """

#     @property
#     def key_source(self):
#         keys = fnn.ModelNetwork & (fnn.Network.VisualNetwork & fnn.Data.VisualScan)
#         return keys.proj()

#     def make(self, key):
#         from foundation.recording.trace import TraceSet

#         # scan data
#         data = fnn.ModelNetwork & key
#         data = fnn.Network.VisualNetwork & data
#         data = fnn.Data.VisualScan & data

#         # units
#         units = recording.ScanUnits & data
#         units = (TraceSet & units).members
#         units = units.fetch(as_dict=True, order_by="traceset_index")

#         # units dataframe
#         df = pd.merge(pd.DataFrame([key]), pd.DataFrame(units), how="cross")
#         df = df.rename(columns={"traceset_index": "response_index"})

#         # insert
#         self.insert(df, ignore_extra_fields=True)


# @schema.computed
# class FnnVisualScanResponse:
#     definition = """
#     -> fnn.ModelNetwork
#     -> recording.Trace.ScanUnit
#     ----
#     response_index              : int unsigned  # response index
#     """

#     @property
#     def key_source(self):
#         keys = fnn.ModelNetwork & (fnn.Network.VisualNetwork & fnn.Data.VisualScan)
#         return keys.proj()

#     def make(self, key):
#         from foundation.recording.trace import TraceSet

#         # scan data
#         data = fnn.ModelNetwork & key
#         data = fnn.Network.VisualNetwork & data
#         data = fnn.Data.VisualScan & data

#         # units
#         units = recording.ScanUnits & data
#         units = (TraceSet & units).members
#         units = units.fetch(as_dict=True, order_by="traceset_index")

#         # units dataframe
#         df = pd.merge(pd.DataFrame([key]), pd.DataFrame(units), how="cross")
#         df = df.rename(columns={"traceset_index": "response_index"})

#         # insert
#         self.insert(df, ignore_extra_fields=True)


# @schema.computed
# class FnnVisualScanTrialResponse:
#     definition = """
#     -> recording.TrialFilterSet
#     -> FnnVisualScanResponse
#     perspective                 : bool          # use recording perspective
#     modulation                  : bool          # use recording modulation
#     ----
#     -> ResponseSet
#     """

#     def make(self, key):
#         pass


# @schema.computed
# class VisualScanFnnTrialResponse:
#     definition = """
#     -> fnn.ModelNetwork
#     -> pipe_fuse.ScanSet.Unit
#     -> FnnTrialResponse
#     ---
#     -> TrialResponse
#     """

#     @property
#     def key_source(self):
#         models = fnn.ModelNetwork & (fnn.Network.VisualNetwork & fnn.Data.VisualScan)
#         return (recording.TrialFilterSet * models).proj()

#     def make(self, key):
#         from foundation.recording.trace import TraceSet

#         # dataset key
#         data = fnn.ModelNetwork & key
#         data = fnn.Network.VisualNetwork & data
#         data = fnn.Data.VisualScan & data

#         # units
#         units = recording.ScanUnits & data
#         units = (TraceSet & units).members
#         units = merge(units, recording.Trace.ScanUnit)
#         units = units.fetch(as_dict=True, order_by="traceset_index")

#         # units dataframe
#         df = pd.merge(pd.DataFrame([key]), pd.DataFrame(units), how="cross")
#         df = df.rename(columns={"traceset_index": "response_index"})

#         # data spec
#         spec = data.proj("rate_id", spec_id="units_id") * fnn.Spec.TraceSpec
#         df["resample_id"], df["offset_id"], df["rate_id"] = spec.fetch1("resample_id", "offset_id", "rate_id")

#         # insert trial response
#         keys = df.to_dict(orient="records")
#         TrialResponse.insert(keys, skip_duplicates=True, ignore_extra_fields=True)

#         # fnn trial keys
#         for perspective in [True, False]:
#             for modulation in [True, False]:

#                 df["trial_perspective"] = perspective
#                 df["trial_modulation"] = modulation
#                 keys = df.to_dict(orient="records")

#                 # insert fnn trial response
#                 FnnTrialResponse.insert(keys, skip_duplicates=True, ignore_extra_fields=True)

#                 # insert computed
#                 self.insert(keys, ignore_extra_fields=True)

#         # fill response
#         Response.fill()
