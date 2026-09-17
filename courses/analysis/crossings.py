"""
Directed, bounded, time-interpolated gate crossing detection (design doc
section 9). Pure functions -- no Django models, no database access -- so
they can be tested against synthetic tracks independent of any flight or
course object.

Replaces the legacy flights/utils/gate_calculator.py, which (per the design
doc's audit, section 2):
  - tested the infinite gate line instead of the finite gate segment,
  - required distance to both markers < 5m for a 10m gate, which is
    impossible by the triangle inequality,
  - fell back to "nearest approach anywhere in the track" with no maximum
    distance when no real crossing existed, silently returning a result.

This module fixes all three: crossings are tested against the gate's
actual finite span, using an exact local projection (no width bug is even
expressible, since the finite-span test is a real geometric containment
check), and "no crossing found" is a distinct, honestly-reported status
from "found a crossing" -- the closest-approach diagnostic is reported
separately and is never substituted as a successful entry.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from courses.geometry import LocalProjection

LOW_CONFIDENCE_GAP_S = 0.5
AMBIGUOUS_GAP_S = 1.0


@dataclass
class GateFrame:
    """A gate's local geometry, precomputed once per gate for repeated use
    against many track segments."""
    gate_id: str
    projection: LocalProjection
    half_width_m: float
    # Unit vectors in the gate's own local (east, north)-like frame,
    # derived from the gate's own stored endpoints so this works
    # identically for straight and curved-course gates without needing any
    # separately stored heading/tangent.
    forward: tuple
    tangent: tuple  # points toward "left" (positive lateral_offset)

    @classmethod
    def from_gate(cls, gate: Dict) -> 'GateFrame':
        left = gate['left_endpoint']
        right = gate['right_endpoint']
        center_lon = (left['lon'] + right['lon']) / 2
        center_lat = (left['lat'] + right['lat']) / 2
        proj = LocalProjection.at(center_lon, center_lat)

        lx, ly = proj.to_local(left['lon'], left['lat'])
        length = (lx ** 2 + ly ** 2) ** 0.5
        if length == 0:
            raise ValueError(f"Gate {gate.get('id')} has a zero-length span (left == right)")
        tangent = (lx / length, ly / length)
        # forward = tangent rotated -90 degrees (clockwise), matching the
        # (forward, left) convention in courses.geometry: left = rotate90_ccw(forward).
        forward = (tangent[1], -tangent[0])

        return cls(
            gate_id=gate.get('id', ''),
            projection=proj,
            half_width_m=length,  # lx,ly is the vector to the LEFT endpoint from center, i.e. half the width
            forward=forward,
            tangent=tangent,
        )

    def local_xy(self, lon: float, lat: float):
        x, y = self.projection.to_local(lon, lat)
        return x * self.forward[0] + y * self.forward[1], x * self.tangent[0] + y * self.tangent[1]


@dataclass
class GateCrossingResult:
    gate_id: str
    status: str  # 'crossed' | 'not_crossed'
    within_span: Optional[bool] = None

    crossing_index: Optional[int] = None  # index of the sample AFTER the crossing (p1)
    alpha: Optional[float] = None  # fractional position between p0 (0.0) and p1 (1.0)
    crossing_time: Optional[object] = None
    crossing_lon: Optional[float] = None
    crossing_lat: Optional[float] = None
    lateral_offset_m: Optional[float] = None  # signed position along the gate span; 0 = dead center, +/- = toward left/right endpoint
    speed_mps: Optional[float] = None
    heading_deg: Optional[float] = None
    altitude_agl: Optional[float] = None  # estimated height AGL at the crossing -- GPS-derived, see vertical_accuracy_m
    vertical_accuracy_m: Optional[float] = None  # receiver-reported accuracy at the crossing, not a verified error bound

    sample_gap_s: Optional[float] = None
    confidence: str = 'unknown'  # 'ok' | 'low_confidence_gap' | 'ambiguous_gap'

    all_forward_crossing_indices: List[int] = field(default_factory=list)
    reverse_crossing_indices: List[int] = field(default_factory=list)

    closest_approach_m: Optional[float] = None
    closest_approach_index: Optional[int] = None

    warnings: List[str] = field(default_factory=list)


def _interp(a, b, alpha):
    if a is None or b is None:
        return None
    return a + alpha * (b - a)


def seconds_between(t0, t1) -> float:
    """Real flight GPS points store timestamp as a Unix epoch float (see
    flights/flight_manager.py); tests and other callers may use datetime.
    Support both rather than assuming one representation."""
    delta = t1 - t0
    return delta.total_seconds() if hasattr(delta, 'total_seconds') else float(delta)


def analyze_gate_crossing(points: List[Dict], gate: Dict, reference_index: Optional[int] = None) -> GateCrossingResult:
    """Find where a track crosses a gate's finite span, in the forward
    direction of travel, with time-interpolated position/speed/heading.

    points: chronological list of dicts with at least 'timestamp', 'lat',
    'lon'; optionally 'ground_speed', 'heading', 'altitude_agl' for
    interpolation.
    gate: a gate record as produced by courses.geometry (needs
    'left_endpoint', 'right_endpoint'; 'id' optional).
    reference_index: when the track crosses this gate's plane more than
    once (e.g. a DZ landing pattern happens to cross the gate's extended
    line at altitude, long before the actual swoop), pick the forward
    crossing whose index is closest to this one -- typically the flight's
    flare_idx, mapped into this points list -- instead of just the first
    chronologically. This is a context-based tiebreak (design doc section
    9: "ask for a selection rather than choosing the one with the best
    score"), not a score-maximizing choice.

    Never substitutes a closest-approach for a real crossing -- that's
    reported separately and only as a diagnostic.
    """
    frame = GateFrame.from_gate(gate)
    result = GateCrossingResult(gate_id=frame.gate_id, status='not_crossed')

    if len(points) < 2:
        result.warnings.append('Fewer than 2 points; cannot evaluate a crossing')
        return result

    local_xy = [frame.local_xy(p['lon'], p['lat']) for p in points]

    closest_m = None
    closest_idx = None
    forward_candidates = []  # list of (index, alpha, x0, y0, x1, y1)

    for i in range(1, len(points)):
        x0, y0 = local_xy[i - 1]
        x1, y1 = local_xy[i]

        # Closest-approach diagnostic (2D distance to gate center), tracked
        # across every sample regardless of crossing status.
        d0 = (x0 ** 2 + y0 ** 2) ** 0.5
        if closest_m is None or d0 < closest_m:
            closest_m, closest_idx = d0, i - 1

        if x0 == 0.0 or x1 == 0.0:
            # Exact-on-plane touch. Treat as a degenerate forward crossing
            # only if the segment is otherwise moving forward (x1 > x0);
            # avoid double-counting by only firing on the p1==0 case (the
            # next segment's p0==0 would otherwise double-report it).
            if x1 == 0.0 and x1 > x0:
                forward_candidates.append((i, 0.0 if x0 == 0.0 else 1.0, x0, y0, x1, y1))
            continue

        if x0 * x1 < 0:
            if x1 > x0:
                # Forward crossing (upstream/negative -> downstream/positive)
                alpha = -x0 / (x1 - x0)
                forward_candidates.append((i, alpha, x0, y0, x1, y1))
            else:
                result.reverse_crossing_indices.append(i)

    result.all_forward_crossing_indices = [c[0] for c in forward_candidates]

    # Closest approach on the last point too (loop only visits segments).
    x_last, y_last = local_xy[-1]
    d_last = (x_last ** 2 + y_last ** 2) ** 0.5
    if closest_m is None or d_last < closest_m:
        closest_m, closest_idx = d_last, len(points) - 1

    result.closest_approach_m = round(closest_m, 4) if closest_m is not None else None
    result.closest_approach_index = closest_idx

    if not forward_candidates:
        if result.reverse_crossing_indices:
            result.warnings.append(
                f'Track crossed gate {frame.gate_id} only backwards '
                f'({len(result.reverse_crossing_indices)}x); not treated as an entry'
            )
        return result

    if reference_index is not None and len(forward_candidates) > 1:
        selected = min(forward_candidates, key=lambda c: abs(c[0] - reference_index))
    else:
        selected = forward_candidates[0]

    i, alpha, x0, y0, x1, y1 = selected
    p0, p1 = points[i - 1], points[i]

    lateral_offset = _interp(y0, y1, alpha)
    result.status = 'crossed'
    result.crossing_index = i
    result.alpha = round(alpha, 6)
    result.lateral_offset_m = round(lateral_offset, 4)
    result.within_span = abs(lateral_offset) <= frame.half_width_m

    result.crossing_lon = _interp(p0['lon'], p1['lon'], alpha)
    result.crossing_lat = _interp(p0['lat'], p1['lat'], alpha)

    t0, t1 = p0['timestamp'], p1['timestamp']
    dt = seconds_between(t0, t1)
    result.sample_gap_s = round(dt, 3)
    result.crossing_time = t0 + (t1 - t0) * alpha

    speed = _interp(p0.get('ground_speed'), p1.get('ground_speed'), alpha)
    heading = _interp(p0.get('heading'), p1.get('heading'), alpha)
    altitude = _interp(p0.get('altitude_agl'), p1.get('altitude_agl'), alpha)
    v_acc = _interp(p0.get('v_acc'), p1.get('v_acc'), alpha)
    result.speed_mps = round(speed, 3) if speed is not None else None
    result.heading_deg = round(heading, 2) if heading is not None else None
    result.altitude_agl = round(altitude, 2) if altitude is not None else None
    result.vertical_accuracy_m = round(v_acc, 2) if v_acc is not None else None

    if dt > AMBIGUOUS_GAP_S:
        result.confidence = 'ambiguous_gap'
        result.warnings.append(
            f'{dt:.2f}s gap around the crossing; do not auto-classify this as a valid crossing'
        )
    elif dt > LOW_CONFIDENCE_GAP_S:
        result.confidence = 'low_confidence_gap'
        result.warnings.append(f'{dt:.2f}s gap around the crossing; treat with reduced confidence')
    else:
        result.confidence = 'ok'

    if not result.within_span:
        result.warnings.append(
            f'Crossed the gate line {abs(lateral_offset):.2f}m from center, outside the '
            f'{frame.half_width_m:.2f}m half-width span -- out-of-span, not a successful gate entry'
        )

    if len(result.all_forward_crossing_indices) > 1:
        selection_note = (
            'closest to the reference index' if reference_index is not None
            else 'the first chronologically'
        )
        result.warnings.append(
            f'{len(result.all_forward_crossing_indices)} forward crossings found; '
            f'reporting {selection_note} -- caller should confirm approach selection'
        )

    return result


def analyze_course_crossings(points: List[Dict], gates: List[Dict],
                              reference_index: Optional[int] = None) -> Dict[str, GateCrossingResult]:
    """Evaluate every gate in a course against the same track. Does not
    enforce gate order or eligibility -- that belongs to a rules profile,
    layered on top of these raw geometric results (design doc section 9).

    reference_index is passed through to each gate's crossing selection
    (see analyze_gate_crossing) -- typically the flight's flare_idx mapped
    into this points list, so all gates prefer crossings near the actual
    swoop rather than an earlier incidental pass near the course."""
    return {
        gate.get('id', str(i)): analyze_gate_crossing(points, gate, reference_index=reference_index)
        for i, gate in enumerate(gates)
    }
