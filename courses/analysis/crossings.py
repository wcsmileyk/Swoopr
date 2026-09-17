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


def analyze_gate_crossing(points: List[Dict], gate: Dict) -> GateCrossingResult:
    """Find where a track crosses a gate's finite span, in the forward
    direction of travel, with time-interpolated position/speed/heading.

    points: chronological list of dicts with at least 'timestamp', 'lat',
    'lon'; optionally 'ground_speed', 'heading', 'altitude_agl' for
    interpolation.
    gate: a gate record as produced by courses.geometry (needs
    'left_endpoint', 'right_endpoint'; 'id' optional).

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
    first_forward = None

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
                result.all_forward_crossing_indices.append(i)
                if first_forward is None:
                    first_forward = (i, 0.0 if x0 == 0.0 else 1.0, x0, y0, x1, y1)
            continue

        if x0 * x1 < 0:
            if x1 > x0:
                # Forward crossing (upstream/negative -> downstream/positive)
                result.all_forward_crossing_indices.append(i)
                if first_forward is None:
                    alpha = -x0 / (x1 - x0)
                    first_forward = (i, alpha, x0, y0, x1, y1)
            else:
                result.reverse_crossing_indices.append(i)

    # Closest approach on the last point too (loop only visits segments).
    x_last, y_last = local_xy[-1]
    d_last = (x_last ** 2 + y_last ** 2) ** 0.5
    if closest_m is None or d_last < closest_m:
        closest_m, closest_idx = d_last, len(points) - 1

    result.closest_approach_m = round(closest_m, 4) if closest_m is not None else None
    result.closest_approach_index = closest_idx

    if first_forward is None:
        if result.reverse_crossing_indices:
            result.warnings.append(
                f'Track crossed gate {frame.gate_id} only backwards '
                f'({len(result.reverse_crossing_indices)}x); not treated as an entry'
            )
        return result

    i, alpha, x0, y0, x1, y1 = first_forward
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
        result.warnings.append(
            f'{len(result.all_forward_crossing_indices)} forward crossings found; '
            f'reporting the first chronologically -- caller should confirm approach selection'
        )

    return result


def analyze_course_crossings(points: List[Dict], gates: List[Dict]) -> Dict[str, GateCrossingResult]:
    """Evaluate every gate in a course against the same track. Does not
    enforce gate order or eligibility -- that belongs to a rules profile,
    layered on top of these raw geometric results (design doc section 9)."""
    return {gate.get('id', str(i)): analyze_gate_crossing(points, gate) for i, gate in enumerate(gates)}
