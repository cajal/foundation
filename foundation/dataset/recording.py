from djutils import merge, row_property
from foundation.recording import trial, trace, scan
from foundation.schemas import dataset as schema


# @schema.lookup
# class TraceDataType:
#     definition = """
#     trace_dtype     : varchar(64)       # trace data type
#     """


# # -------------- Recording --------------

# # -- Recording Base --


# class _Recording:
#     """Recording Dataset"""

#     @row_property
#     def trials_traces(self):
#         """
#         Returns
#         -------
#         foundation.recording.trial.TrialSet
#             single tuple
#         foundation.recording.trace.TraceSet
#             single tuple
#         foundation.dataset.recording.TraceDataType
#             single tuple
#         """
#         raise NotImplementedError()


# # -- Recording Types --


# @schema.lookup
# class ScanUnit(_Recording):
#     definition = """
#     -> scan.ScanTrialSet
#     -> scan.ScanUnitSet
#     """

#     @row_property
#     def trials_traces(self):
#         return (
#             trial.TrialSet & (scan.ScanTrialSet & self),
#             trace.TraceSet & (scan.ScanUnitSet & self),
#             TraceDataType & dict(trace_dtype="neuron"),
#         )


# @schema.lookup
# class ScanPerspective(_Recording):
#     definition = """
#     -> scan.ScanTrialSet
#     -> scan.ScanPerspectiveSet
#     """

#     @row_property
#     def trials_traces(self):
#         return (
#             trial.TrialSet & (scan.ScanTrialSet & self),
#             trace.TraceSet & (scan.ScanPerspectiveSet & self),
#             TraceDataType & dict(trace_dtype="perspective"),
#         )


# @schema.lookup
# class ScanModulation(_Recording):
#     definition = """
#     -> scan.ScanTrialSet
#     -> scan.ScanModulationSet
#     """

#     @row_property
#     def trials_traces(self):
#         return (
#             trial.TrialSet & (scan.ScanTrialSet & self),
#             trace.TraceSet & (scan.ScanModulationSet & self),
#             TraceDataType & dict(trace_dtype="modulation"),
#         )


# # -- Recording Links --


# @schema.link
# class RecordingLink:
#     links = [ScanUnit, ScanPerspective, ScanModulation]
#     name = "recording"
#     comment = "recording dataset"


# @schema.set
# class RecordingSet:
#     keys = [RecordingLink]
#     name = "recordings"
#     comment = "set of recording datasets"


# # -- Computed Recording --


# @schema.computed
# class RecordingData:
#     definition = """
#     -> RecordingLink
#     ---
#     -> trial.TrialSet
#     -> trace.TraceSet
#     -> TraceDataType
#     """

#     def make(self, key):
#         trials, traces, dtype = (RecordingLink & key).link.trials_traces

#         key["trials_id"] = trials.fetch1("trials_id")
#         key["traces_id"] = traces.fetch1("traces_id")
#         key["trace_dtype"] = dtype.fetch1("trace_dtype")

#         self.insert1(key)
