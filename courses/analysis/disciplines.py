"""
Discipline-specific metrics derived from raw gate-crossing results. Kept
separate from the crossing engine itself (design doc section 9: "the
scoring engine consumes checks/evidence after geometry, not inside the
parser") and from any rules-profile/scoring concerns, which are a later
phase -- this module reports plain measured facts (elapsed time, speed,
down-course distance), never a score or a pass/fail eligibility call.
"""

from typing import Dict, List, Optional

from courses.analysis.crossings import GateCrossingResult, seconds_between
from courses.geometry import EntrySetup


def _entry_setup_from_revision(course_revision) -> EntrySetup:
    return EntrySetup(
        g1_lon=course_revision.g1_lon,
        g1_lat=course_revision.g1_lat,
        entry_heading_deg_true=course_revision.entry_heading_deg_true,
        carve_direction=course_revision.carve_direction or 'left',
        course_width_m=course_revision.course_width_m,
    )


def compute_distance_metrics(gate_results: Dict[str, GateCrossingResult], window_points: List[Dict],
                              course_revision) -> Dict:
    metrics = {}
    g1 = gate_results.get('distance:G1')
    g5 = gate_results.get('distance:G5')

    if g1 and g1.status == 'crossed':
        metrics['entry_speed_mps'] = round(g1.speed_mps, 2) if g1.speed_mps is not None else None
        metrics['entry_within_span'] = g1.within_span

    if g5 and g5.status == 'crossed':
        metrics['g5_speed_mps'] = round(g5.speed_mps, 2) if g5.speed_mps is not None else None
        metrics['g5_within_span'] = g5.within_span
        if g1 and g1.status == 'crossed':
            metrics['elapsed_g1_to_g5_s'] = round(seconds_between(g1.crossing_time, g5.crossing_time), 2)

    if window_points:
        setup = _entry_setup_from_revision(course_revision)
        last = window_points[-1]
        # The actual "distance" result: the last GPS sample's position
        # projected onto the course's forward direction from G1 -- not the
        # raw east/north offset (which only equals down-course distance
        # when the entry heading happens to be due north) and not a
        # confirmed touchdown/contact point (design doc section 9: "not
        # distance flown or final resting position" -- this raw GPS
        # position is itself only a proxy pending independent evidence).
        x, _y = setup.to_forward_left(last['lon'], last['lat'])
        metrics['down_course_distance_m'] = round(x, 1)

    return metrics


def compute_speed_metrics(gate_results: Dict[str, GateCrossingResult], window_points: List[Dict],
                           course_revision) -> Dict:
    metrics = {}
    ordered = [gate_results.get(f'speed:G{i}') for i in range(1, 6)]

    if all(g is not None and g.status == 'crossed' for g in ordered):
        times = [g.crossing_time for g in ordered]
        metrics['elapsed_g1_to_g5_s'] = round(seconds_between(times[0], times[-1]), 2)
        metrics['sector_times_s'] = [
            round(seconds_between(times[i], times[i + 1]), 2) for i in range(len(times) - 1)
        ]
        metrics['gate_speeds_mps'] = [round(g.speed_mps, 2) if g.speed_mps is not None else None for g in ordered]
        metrics['all_gates_within_span'] = all(g.within_span for g in ordered)
    elif ordered[0] is not None and ordered[0].status == 'crossed':
        metrics['entry_speed_mps'] = round(ordered[0].speed_mps, 2) if ordered[0].speed_mps is not None else None

    return metrics


def compute_accuracy_metrics(gate_results: Dict[str, GateCrossingResult], window_points: List[Dict],
                              course_revision) -> Dict:
    # Water-contact evidence and landing-zone scoring require independent
    # observation beyond GPS (design doc sections 4, 9) -- not implemented
    # here. This reports only what's geometrically knowable: which water
    # gates the track actually passed within span.
    metrics = {}
    gates_crossed = []
    for i in range(1, 5):
        g = gate_results.get(f'accuracy:G{i}')
        if g is not None and g.status == 'crossed' and g.within_span:
            gates_crossed.append(i)
    metrics['water_gates_crossed_in_span'] = gates_crossed
    return metrics


DISCIPLINE_METRIC_FUNCS = {
    'distance': compute_distance_metrics,
    'speed': compute_speed_metrics,
    'accuracy': compute_accuracy_metrics,
}


def compute_discipline_metrics(discipline: str, gate_results: Dict[str, GateCrossingResult],
                                window_points: List[Dict], course_revision) -> Dict:
    func = DISCIPLINE_METRIC_FUNCS.get(discipline)
    if func is None:
        return {}
    return func(gate_results, window_points, course_revision)


def entry_gate_id_for(discipline: str) -> str:
    return f'{discipline}:G1'
