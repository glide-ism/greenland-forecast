"""
Evaluation-time scheduling for the forward model.

Observations carry calendar-year timestamps, and the forward model must emit
state exactly at those times. `build_step_sequence` designs the sequence of
(time, dt) steps: a uniform spinup grid at the configured dt, with extra
breakpoints snapped onto every required observation time (variable dt, capped
at the configured maximum). Times are plain Python floats throughout so
snapshot dictionary keys match the requested times exactly — no float32
clock-drift matching.
"""
import math
from typing import Iterable, Optional, Sequence


def merge_times(*iterables: Optional[Iterable[float]], eps: float = 1e-6
                ) -> tuple:
    """Union of time values from any number of (possibly None) iterables,
    sorted ascending and deduplicated within `eps`."""
    times: list = []
    for it in iterables:
        if it is None:
            continue
        times.extend(float(t) for t in it)
    times.sort()
    merged: list = []
    for t in times:
        if not merged or t - merged[-1] > eps:
            merged.append(t)
    return tuple(merged)


def build_step_sequence(
    *,
    t_start: float,
    t_end: float,
    dt_max: float,
    required_times: Sequence[float] = (),
    eps: float = 1e-6,
    dt_schedule: Sequence = (),
) -> list:
    """Design the forward model's step sequence as [(t_next, dt), ...].

    `dt_schedule` = ((t_from, dt), ...) refines the stepping from `t_from`
    onward (later entries override earlier ones): each refined segment is
    anchored at its start, at every required time inside it and at its end,
    and every gap between anchors is split into ceil(gap / dt) EQUAL steps —
    so steps never exceed dt, no slivers appear next to observation epochs,
    and snapshots still land on the exact requested floats. The legacy
    uniform grid applies before the first entry.

    The run covers (t_start, horizon] where horizon = max(t_end, max required
    time) — a required time beyond the nominal calibration end extends the run
    (climate-anomaly lookups clamp to the last available year, so forcing is
    defined). Breakpoints are the uniform grid {t_start + k * dt_max} unioned
    with `required_times`; a grid point within `eps` of a required time is
    replaced by the required time so state is emitted at exactly the requested
    float. Every dt is positive and <= dt_max + eps.

    Raises ValueError if a required time falls at or before t_start (the model
    cannot emit state it never simulates — lower t_start or override the
    observation time), or if the sequence would be empty.
    """
    t_start = float(t_start)
    t_end = float(t_end)
    dt_max = float(dt_max)
    if dt_max <= 0:
        raise ValueError(f"dt_max must be positive, got {dt_max}")

    required = merge_times(required_times, eps=eps)
    too_early = [t for t in required if t <= t_start + eps]
    if too_early:
        raise ValueError(
            f"required time(s) {too_early} fall at or before t_start="
            f"{t_start}; lower t_start (or override the observation time) so "
            f"the model simulates through every observation epoch."
        )

    horizon = max([t_end] + list(required))

    schedule = sorted((float(a), float(b)) for a, b in dt_schedule)
    for _, d in schedule:
        if d <= 0:
            raise ValueError(f"dt_schedule steps must be positive, got {schedule}")
    refine_from = schedule[0][0] if schedule and schedule[0][0] < horizon - eps else None

    # Uniform grid, skipping points that collide (within eps) with a required
    # time — the required float wins so snapshot keys are exact. With a
    # schedule the uniform grid stops at the first refinement start.
    breakpoints = list(required)
    coarse_end = horizon if refine_from is None else refine_from
    k = 1
    while True:
        t = t_start + k * dt_max
        if t > coarse_end - eps:
            break
        if all(abs(t - r) > eps for r in required):
            breakpoints.append(t)
        k += 1
    if refine_from is not None:
        if refine_from > t_start + eps and all(abs(refine_from - b) > eps for b in breakpoints):
            breakpoints.append(refine_from)
        # Refined segments: anchors = segment bounds + required times inside;
        # each gap split into ceil(gap/dt) equal steps.
        bounds = [s for s, _ in schedule] + [horizon]
        for (seg_start, seg_dt), seg_end in zip(schedule, bounds[1:]):
            seg_start = max(seg_start, t_start)
            if seg_end <= seg_start + eps:
                continue
            anchors = sorted({seg_start, seg_end}
                             | {r for r in required if seg_start + eps < r < seg_end - eps})
            for a, b in zip(anchors[:-1], anchors[1:]):
                n = max(1, int(math.ceil((b - a) / seg_dt - eps)))
                for j in range(1, n):
                    t = a + (b - a) * j / n
                    if all(abs(t - r) > eps for r in required):
                        breakpoints.append(t)
            if all(abs(seg_end - b) > eps for b in breakpoints):
                breakpoints.append(seg_end)
    if all(abs(horizon - b) > eps for b in breakpoints):
        breakpoints.append(horizon)
    breakpoints = sorted(b for b in breakpoints if t_start + eps < b <= horizon + eps)

    steps = []
    t_prev = t_start
    for t_next in breakpoints:
        steps.append((t_next, t_next - t_prev))
        t_prev = t_next
    if not steps:
        raise ValueError(
            f"empty step sequence for t_start={t_start}, t_end={t_end}, "
            f"required_times={required}; the horizon must exceed t_start."
        )
    return steps
