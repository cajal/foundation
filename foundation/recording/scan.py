import numpy as np
from djutils import merge
from foundation.virtual import scan, stimulus
from foundation.virtual.bridge import pipe_stim, pipe_fuse, pipe_eye, pipe_tread, pipe_shared
from foundation.recording.trial import (
    ScanTrial,
    Trial,
    TrialBounds,
    TrialVideo,
    TrialSet,
    TrialFilterSet,
)
from foundation.recording.trace import (
    ScanUnit,
    ScanUnitRaw,
    ScanPupil,
    ScanTreadmill,
    Trace,
    TraceSet,
    TraceFilterSet,
)
from foundation.schemas import recording as schema


@schema.computed
class ScanRecording:
    definition = """
    -> scan.Scan
    ---
    -> TrialSet
    """

    def make(self, key):
        # scan trials
        trials = (pipe_stim.Trial & key).proj()

        # insert trials
        ScanTrial.insert(trials - ScanTrial)
        Trial.fill()

        # trial keys
        trials = Trial & (Trial.ScanTrial & key)

        # trial set
        trials = TrialSet.fill(trials, prompt=False)

        # insert
        self.insert1(dict(key, **trials))


@schema.computed
class ScanVideoTimeScale:
    definition = """
    -> scan.Scan
    ---
    time_scale      : double    # video time scale
    """

    def make(self, key):
        # trials merged with video info
        trials = TrialSet & merge(scan.Scan & key, ScanRecording)
        trials = merge(trials.members, Trial.ScanTrial, TrialBounds, TrialVideo, stimulus.VideoInfo)

        # trial and video timing
        start, end, frames, period = trials.fetch("start", "end", "frames", "period")
        rperiod = (end - start) / (frames - 1)

        # median time scale
        key["time_scale"] = np.nanmedian(rperiod / period)
        self.insert1(key)


@schema.computed
class ScanTrials:
    definition = """
    -> scan.Scan
    -> TrialFilterSet
    ---
    -> TrialSet
    """

    def make(self, key):
        # all trials
        trials = TrialSet & merge(scan.Scan & key, ScanRecording)
        trials = Trial & trials.members

        # filter trials
        trials = (TrialFilterSet & key).filter(trials)

        # trial set
        trials = TrialSet.fill(trials, prompt=False)

        # insert
        self.insert1(dict(key, **trials))


@schema.computed
class ScanUnits:
    definition = """
    -> scan.Scan
    -> pipe_fuse.ScanDone
    -> TraceFilterSet
    ---
    -> TraceSet
    """

    def make(self, key):
        # scan units
        units = pipe_fuse.Activity.Trace & key

        # insert traces
        ScanUnit.insert(units - ScanUnit, ignore_extra_fields=True)
        Trace.fill()

        # trace keys
        traces = Trace & (Trace.ScanUnit & key)

        # filter traces
        traces = (TraceFilterSet & key).filter(traces)

        # trace set
        traces = TraceSet.fill(traces, prompt=False)

        # insert
        self.insert1(dict(key, **traces))

@schema.computed
class ScanUnitsRaw:
    definition = """
    -> scan.Scan
    -> pipe_shared.PipelineVersion
    -> pipe_shared.SegmentationMethod
    -> TraceFilterSet
    ---
    -> TraceSet
    """
    
    @property
    def key_source(self):
        keys = (scan.Scan * pipe_shared.PipelineVersion * pipe_shared.SegmentationMethod * TraceFilterSet).proj()
        return keys & pipe_fuse.ScanDone

    def make(self, key):
        # scan units
        units = pipe_fuse.ScanSet.Unit & key

        # insert traces
        ScanUnit.insert(units - ScanUnit, ignore_extra_fields=True)
        Trace.fill()

        # trace keys
        traces = Trace & (Trace.ScanUnit & key)

        # filter traces
        traces = (TraceFilterSet & key).filter(traces)

        # trace set
        traces = TraceSet.fill(traces, prompt=False)

        # insert
        self.insert1(dict(key, **traces))


@schema.computed
class ScanVisualPerspectives:
    definition = """
    -> scan.Scan
    -> pipe_eye.FittedPupil
    ---
    -> TraceSet
    """

    def make(self, key):
        # scan pupil traces
        pupils = merge(scan.Scan & key, scan.PupilTrace & key).proj()
        pupils &= [dict(pupil_type="center_x"), dict(pupil_type="center_y")]

        # insert traces
        ScanPupil.insert(pupils - ScanPupil)
        Trace.fill()

        # trace keys
        traces = Trace.ScanPupil & pupils

        # trace set
        traces = TraceSet.fill(traces, prompt=False)

        # insert
        self.insert1(dict(key, **traces))


@schema.computed
class ScanVisualModulations:
    definition = """
    -> scan.Scan
    -> pipe_eye.FittedPupil
    ---
    -> TraceSet
    """

    def make(self, key):
        # scan pupil trace
        pupil = merge(scan.Scan & key, scan.PupilTrace & key).proj() & dict(pupil_type="radius")

        # scan treadmill trace
        tread = merge(scan.Scan & key, pipe_tread.Treadmill).proj()

        # insert traces
        ScanPupil.insert(pupil - ScanPupil)
        ScanTreadmill.insert(tread - ScanTreadmill)
        Trace.fill()

        # trace keys
        traces = [Trace.ScanPupil & pupil, Trace.ScanTreadmill & tread]

        # trace set
        traces = TraceSet.fill(traces, prompt=False)

        # insert
        self.insert1(dict(key, **traces))


@schema.computed
class ScanUnitOrder:
    definition = """
    -> ScanUnits
    -> Trace
    ---
    trace_order     : int unsigned  # trace order
    """

    @property
    def key_source(self):
        return ScanUnits.proj()

    def make(self, key):
        # trace set
        traces = ScanUnits & key
        traces = (TraceSet & traces).members
        traces = traces * Trace.ScanUnit

        # fetch trace ids in order
        trace_ids = traces.fetch("trace_id", order_by=ScanUnit.primary_key)

        # trace keys
        keys = [dict(key, trace_id=t, trace_order=i) for i, t in enumerate(trace_ids)]

        # insert
        self.insert(keys)


@schema.computed
class ScanVisualPerspectiveOrder:
    definition = """
    -> ScanVisualPerspectives
    -> Trace
    ---
    trace_order     : int unsigned  # trace order
    """

    @property
    def key_source(self):
        return ScanVisualPerspectives.proj()

    def make(self, key):
        # trace set
        traces = ScanVisualPerspectives & key
        traces = (TraceSet & traces).members
        traces = traces * Trace.ScanPupil

        # fetch trace ids in order
        trace_ids = traces.fetch("trace_id", order_by=ScanPupil.primary_key)

        # trace keys
        keys = [dict(key, trace_id=t, trace_order=i) for i, t in enumerate(trace_ids)]

        # insert
        self.insert(keys)


@schema.computed
class ScanVisualModulationOrder:
    definition = """
    -> ScanVisualModulations
    -> Trace
    ---
    trace_order     : int unsigned  # trace order
    """

    @property
    def key_source(self):
        return ScanVisualModulations.proj()

    def make(self, key):
        # trace set
        traces = ScanVisualModulations & key
        traces = (TraceSet & traces).members

        # fetch trace ids
        trace_p = (traces & Trace.ScanPupil).fetch1("trace_id")
        trace_t = (traces & Trace.ScanTreadmill).fetch1("trace_id")
        trace_ids = [trace_p, trace_t]

        # trace keys
        keys = [dict(key, trace_id=t, trace_order=i) for i, t in enumerate(trace_ids)]

        # insert
        self.insert(keys)
