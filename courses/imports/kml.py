"""
KML import for the course system (design doc section 6).

Scope for this first pass: "explicit geometry" mode only -- a KML that
already lays out every gate marker (the two supplied eastern fixtures).
G1-seed-only KMLs and gSwoop survey import are separate, later slices.

Security (per doc section 6): parse with defusedxml (rejects external
entities/DTD), never fetch NetworkLinks or remote assets, and bound file
size and element/coordinate counts before doing any real work.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from defusedxml import ElementTree as DefusedET

from courses.geometry import (
    DEFAULT_COURSE_WIDTH_M,
    EntrySetup,
    GeometryError,
    LocalProjection,
    generate_geometry,
    haversine_m,
)

KML_NS = 'http://www.opengis.net/kml/2.2'

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_COORDINATE_TUPLES = 100_000
MAX_ELEMENT_DEPTH = 32

# Reconciling a point-defined gate against its own "G{n} Gate" line, if
# present, is a file-internal consistency check -- not survey accuracy.
LINE_POINT_RECONCILE_TOLERANCE_M = 0.20

# Comparing imported (explicit) coordinates against the canonical generator
# is a computational-compatibility check for the rounded-radius legacy
# fixtures, not a claim about physical placement accuracy.
TEMPLATE_COMPARISON_TOLERANCE_M = 0.05

_GATE_MARKER_RE = re.compile(r'^G(\d+)\s+(Inside|Outside|Center)$', re.IGNORECASE)
_GATE_LINE_RE = re.compile(r'^G(\d+)\s+Gate$', re.IGNORECASE)

_DISCIPLINE_BY_GATE_NUMBERS = {
    frozenset({1, 5}): 'distance',
    frozenset({1, 2, 3, 4, 5}): 'speed',
    frozenset({1, 2, 3, 4}): 'accuracy',
}


class KmlImportError(ValueError):
    """Raised for malformed, oversized, or unrecognized KML input."""


@dataclass
class GateMarkerSet:
    gate_number: int
    inside: Optional[Tuple[float, float]] = None
    outside: Optional[Tuple[float, float]] = None
    center: Optional[Tuple[float, float]] = None


@dataclass
class ParsedKml:
    course_name: str
    description: str
    gates: Dict[int, GateMarkerSet]
    warnings: List[str] = field(default_factory=list)


def _strip_ns(tag: str) -> str:
    return tag.split('}', 1)[-1] if '}' in tag else tag


def _parse_coordinates_text(text: str) -> List[Tuple[float, float]]:
    """KML coordinates are 'lon,lat[,alt] lon,lat[,alt] ...'."""
    points = []
    for tuple_str in text.split():
        parts = tuple_str.split(',')
        if len(parts) < 2:
            raise KmlImportError(f'Malformed coordinate tuple: {tuple_str!r}')
        try:
            lon, lat = float(parts[0]), float(parts[1])
        except ValueError:
            raise KmlImportError(f'Non-numeric coordinate tuple: {tuple_str!r}')
        if not (-180.0 <= lon <= 180.0) or not (-90.0 <= lat <= 90.0):
            raise KmlImportError(f'Coordinate out of range: {tuple_str!r}')
        points.append((lon, lat))
    return points


def parse_kml_bytes(data: bytes) -> ParsedKml:
    if len(data) > MAX_FILE_BYTES:
        raise KmlImportError(f'File exceeds {MAX_FILE_BYTES} byte limit')

    if b'<!DOCTYPE' in data or b'<!ENTITY' in data or b'NetworkLink' in data:
        raise KmlImportError('DTDs, entities, and NetworkLink are not supported')

    try:
        # forbid_dtd also rejects internal/external entity declarations.
        root = DefusedET.fromstring(data, forbid_dtd=True, forbid_entities=True, forbid_external=True)
    except Exception as exc:
        raise KmlImportError(f'Could not parse XML: {exc}') from exc

    def depth(el, current=0):
        if current > MAX_ELEMENT_DEPTH:
            raise KmlImportError('Element nesting too deep')
        for child in el:
            depth(child, current + 1)
    depth(root)

    document = None
    for el in root:
        if _strip_ns(el.tag) == 'Document':
            document = el
            break
    if document is None:
        raise KmlImportError('No <Document> element found')

    course_name = ''
    description = ''
    gates: Dict[int, GateMarkerSet] = {}
    warnings: List[str] = []
    total_coord_tuples = 0

    def gate(n: int) -> GateMarkerSet:
        return gates.setdefault(n, GateMarkerSet(gate_number=n))

    def walk_placemarks(el):
        nonlocal course_name, description, total_coord_tuples
        for child in el:
            tag = _strip_ns(child.tag)
            if tag == 'name' and el is document:
                course_name = (child.text or '').strip()
            elif tag == 'description' and el is document:
                description = (child.text or '').strip()
            elif tag == 'Folder':
                walk_placemarks(child)
            elif tag == 'Placemark':
                _handle_placemark(child)

    def _handle_placemark(placemark):
        nonlocal total_coord_tuples
        name = ''
        geom_points: List[Tuple[float, float]] = []
        for child in placemark:
            tag = _strip_ns(child.tag)
            if tag == 'name':
                name = (child.text or '').strip()
            elif tag in ('Point', 'LineString'):
                for gc in child:
                    if _strip_ns(gc.tag) == 'coordinates' and gc.text:
                        pts = _parse_coordinates_text(gc.text)
                        total_coord_tuples += len(pts)
                        if total_coord_tuples > MAX_COORDINATE_TUPLES:
                            raise KmlImportError('Too many coordinate tuples')
                        geom_points = pts

        if not name or not geom_points:
            return

        marker_match = _GATE_MARKER_RE.match(name)
        line_match = _GATE_LINE_RE.match(name)

        if marker_match:
            n = int(marker_match.group(1))
            role = marker_match.group(2).lower()
            g = gate(n)
            existing = getattr(g, role)
            if existing is not None and existing != geom_points[0]:
                warnings.append(f'Duplicate/conflicting {name!r} placemark ignored')
                return
            setattr(g, role, geom_points[0])
        elif line_match:
            n = int(line_match.group(1))
            if len(geom_points) != 2:
                warnings.append(f'{name!r} line does not have exactly 2 points; skipped for reconciliation')
                return
            g = gate(n)
            _reconcile_gate_line(g, geom_points, warnings)
        # else: boundary/centerline/other reference geometry -- not a gate,
        # intentionally ignored for gate extraction (design doc section 6).

    walk_placemarks(document)

    if not gates:
        raise KmlImportError('No recognizable gate markers found (expected names like "G1 Inside")')

    return ParsedKml(course_name=course_name, description=description, gates=gates, warnings=warnings)


def _reconcile_gate_line(gate: GateMarkerSet, line_points: List[Tuple[float, float]], warnings: List[str]):
    a, b = line_points
    candidates = [p for p in (gate.inside, gate.outside) if p is not None]
    if len(candidates) < 2:
        return
    # Match line endpoints to inside/outside by nearest, then check tolerance.
    pairs = [(gate.inside, a), (gate.outside, b)] if (
        haversine_m(*gate.inside, *a) + haversine_m(*gate.outside, *b)
        <= haversine_m(*gate.inside, *b) + haversine_m(*gate.outside, *a)
    ) else [(gate.inside, b), (gate.outside, a)]
    for point, line_end in pairs:
        d = haversine_m(*point, *line_end)
        if d > LINE_POINT_RECONCILE_TOLERANCE_M:
            warnings.append(
                f'G{gate.gate_number} gate line endpoint is {d:.3f}m from its point '
                f'placemark (tolerance {LINE_POINT_RECONCILE_TOLERANCE_M}m)'
            )


def detect_discipline(parsed: ParsedKml) -> Optional[str]:
    gate_numbers = frozenset(parsed.gates.keys())
    return _DISCIPLINE_BY_GATE_NUMBERS.get(gate_numbers)


def _resolve_inside_outside(g: GateMarkerSet) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    if g.inside is not None and g.outside is not None:
        return g.inside, g.outside
    return None


def derive_entry_setup(parsed: ParsedKml, carve_direction: str, entry_heading_deg_true: Optional[float] = None,
                        course_width_m: Optional[float] = None) -> Tuple[EntrySetup, List[str]]:
    """Derive the entry setup (G1 center, heading, width) from parsed gate
    markers. carve_direction must be supplied by the caller/wizard -- it
    cannot be inferred from geometry alone (design doc section 6).

    Inside/outside meaning follows directly from carve direction: the
    inside of a left-hand turn is the pilot's left, and the inside of a
    right-hand turn is the pilot's right. For a straight course (Distance),
    this labeling is inherited from whichever Speed setup shares the same
    G1, but doesn't affect the straight geometry itself.
    """
    warnings: List[str] = []

    if 1 not in parsed.gates:
        raise KmlImportError('No G1 marker found; cannot derive an entry setup')
    g1 = parsed.gates[1]
    io = _resolve_inside_outside(g1)

    if io is not None:
        inside, outside = io
        derived_width = haversine_m(*inside, *outside)
        width = course_width_m if course_width_m is not None else derived_width
        if course_width_m is not None and abs(derived_width - course_width_m) > TEMPLATE_COMPARISON_TOLERANCE_M:
            warnings.append(
                f'Supplied course_width_m={course_width_m} differs from the G1 marker '
                f'spacing of {derived_width:.3f}m by more than {TEMPLATE_COMPARISON_TOLERANCE_M}m'
            )

        if g1.center is not None:
            g1_lon = (inside[0] + outside[0]) / 2
            g1_lat = (inside[1] + outside[1]) / 2
            d = haversine_m(g1_lon, g1_lat, *g1.center)
            if d > TEMPLATE_COMPARISON_TOLERANCE_M:
                warnings.append(
                    f'G1 Center marker is {d:.3f}m from the Inside/Outside midpoint; using the midpoint'
                )
            g1_center = (g1_lon, g1_lat)
        else:
            g1_center = ((inside[0] + outside[0]) / 2, (inside[1] + outside[1]) / 2)

        heading = entry_heading_deg_true
        if heading is None:
            # The G1 gate line's perpendicular IS the entry heading, by
            # construction (see courses.geometry), for both straight and
            # curved courses -- unlike the chord to a distant gate (e.g. G5 on
            # a 75-degree arc), which is offset from the true tangent heading
            # by roughly half the arc angle. carve_direction resolves which
            # physical marker is "left" vs "right" and so which perpendicular
            # direction is forward: no other gate's position is needed.
            bearing = _local_bearing_deg(inside, outside)
            heading = (bearing - 90) % 360 if carve_direction == 'left' else (bearing + 90) % 360
    else:
        # G1 seed mode (design doc section 6, "G1 center Point"): only a
        # single G1 marker exists (Center, or a lone Inside/Outside point
        # treated as the center) -- there's no measured width or gate line
        # to derive heading from, so both must be supplied explicitly.
        seed_point = g1.center or g1.inside or g1.outside
        if seed_point is None:
            raise KmlImportError('G1 has no usable coordinates at all')
        if entry_heading_deg_true is None:
            raise KmlImportError(
                'G1 has only a single point (no Inside/Outside pair); '
                'supply an explicit entry heading to seed a course from it'
            )
        g1_center = seed_point
        heading = entry_heading_deg_true
        if course_width_m is None:
            width = DEFAULT_COURSE_WIDTH_M
            warnings.append(f'No measured G1 width available; using the default {DEFAULT_COURSE_WIDTH_M}m')
        else:
            width = course_width_m

    setup = EntrySetup(
        g1_lon=g1_center[0], g1_lat=g1_center[1],
        entry_heading_deg_true=heading,
        carve_direction=carve_direction,
        course_width_m=width,
    )
    return setup, warnings


def _local_bearing_deg(origin: Tuple[float, float], point: Tuple[float, float]) -> float:
    """True bearing from origin to point, via the exact local AEQD
    projection centered at origin.

    Deliberately not a spherical great-circle bearing formula: the
    standard sin/cos great-circle formula suffers catastrophic
    cancellation for points only meters apart (like adjacent gate
    markers), losing on the order of 0.1 degrees of precision -- enough
    to put a 70m course tens of centimeters off. AEQD gives exact local
    east/north offsets directly, with no such cancellation.
    """
    import math
    proj = LocalProjection.at(*origin)
    x, y = proj.to_local(*point)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def build_preview(parsed: ParsedKml, carve_direction: str, disciplines: Optional[List[str]] = None,
                   entry_heading_deg_true: Optional[float] = None, course_width_m: Optional[float] = None) -> Dict:
    """Build the full preview dict for CourseImport.extracted: the derived
    entry setup, generated (canonical) geometry per discipline, the
    imported geometry actually found in the file, and comparison
    warnings between the two.

    entry_heading_deg_true/course_width_m are required for a G1-seed file
    (only a single G1 point, no Inside/Outside pair) and optional
    overrides/validation inputs for an explicit-geometry file.
    """
    detected = detect_discipline(parsed)
    if disciplines is None:
        if detected is None:
            raise KmlImportError(
                f'Could not determine discipline from gate numbers {sorted(parsed.gates)}; '
                f'specify disciplines explicitly'
            )
        disciplines = [detected]

    setup, setup_warnings = derive_entry_setup(
        parsed, carve_direction=carve_direction,
        entry_heading_deg_true=entry_heading_deg_true, course_width_m=course_width_m,
    )
    warnings = list(parsed.warnings) + list(setup_warnings)

    try:
        generated = generate_geometry(setup, disciplines)
    except GeometryError as exc:
        raise KmlImportError(str(exc)) from exc

    imported_gates = {}
    for n, g in sorted(parsed.gates.items()):
        io = _resolve_inside_outside(g)
        entry = {'gate_number': n}
        if io is not None:
            entry['inside'] = {'lon': io[0][0], 'lat': io[0][1]}
            entry['outside'] = {'lon': io[1][0], 'lat': io[1][1]}
        if g.center is not None:
            entry['center'] = {'lon': g.center[0], 'lat': g.center[1]}
        imported_gates[n] = entry

    # Compare imported vs. generated gate centers where both exist, per
    # discipline, as a computational-compatibility check.
    comparisons = []
    for discipline, geo in generated.items():
        for gate_record in geo['gates']:
            n = gate_record['gate_number']
            imported = imported_gates.get(n)
            if not imported or 'inside' not in imported:
                continue
            imp_center = (
                (imported['inside']['lon'] + imported['outside']['lon']) / 2,
                (imported['inside']['lat'] + imported['outside']['lat']) / 2,
            )
            gen_center = (
                (gate_record['left_endpoint']['lon'] + gate_record['right_endpoint']['lon']) / 2,
                (gate_record['left_endpoint']['lat'] + gate_record['right_endpoint']['lat']) / 2,
            )
            d = haversine_m(*imp_center, *gen_center)
            comparisons.append({'discipline': discipline, 'gate_number': n, 'deviation_m': round(d, 4)})
            if d > TEMPLATE_COMPARISON_TOLERANCE_M:
                warnings.append(
                    f'{discipline} G{n}: imported gate is {d:.3f}m from the canonical template '
                    f'(tolerance {TEMPLATE_COMPARISON_TOLERANCE_M}m) -- may be a custom, non-standard course'
                )

    # What actually gets committed: for the discipline the file was built
    # for, preserve the imported coordinates (design doc section 6: "An
    # explicit geometry import preserves uploaded positions"). Any other
    # requested discipline seeded from the same G1 uses pure generation --
    # a Speed file's G2-G4 are not reusable as Distance or Accuracy gates.
    final_geometry = {}
    for discipline, geo in generated.items():
        if discipline == detected:
            final_geometry[discipline] = _substitute_imported_gates(
                geo, imported_gates, setup, carve_direction
            )
        else:
            final_geometry[discipline] = geo

    return {
        'course_name': parsed.course_name,
        'description': parsed.description,
        'detected_discipline': detected,
        'disciplines': disciplines,
        'entry_setup': {
            'g1_lon': setup.g1_lon,
            'g1_lat': setup.g1_lat,
            'entry_heading_deg_true': setup.entry_heading_deg_true,
            'course_width_m': setup.course_width_m,
            'carve_direction': setup.carve_direction,
        },
        'imported_gates': imported_gates,
        'generated_geometry': generated,
        'final_geometry': final_geometry,
        'template_comparisons': comparisons,
        'warnings': warnings,
    }


def _substitute_imported_gates(geo: Dict, imported_gates: Dict, setup: EntrySetup, carve_direction: str) -> Dict:
    """Replace each generated gate with its imported coordinates where the
    KML actually provided that gate's Inside/Outside markers, converted to
    our internal left/right convention. Gates the file didn't provide
    (e.g. Accuracy zones, which are never gate markers) keep the generated
    geometry unchanged."""
    proj = LocalProjection.at(setup.g1_lon, setup.g1_lat)
    result = dict(geo)
    new_gates = []
    for gate_record in geo['gates']:
        n = gate_record['gate_number']
        imported = imported_gates.get(n)
        if imported and 'inside' in imported:
            inside = (imported['inside']['lon'], imported['inside']['lat'])
            outside = (imported['outside']['lon'], imported['outside']['lat'])
            left, right = (inside, outside) if carve_direction == 'left' else (outside, inside)
            center_lon = (left[0] + right[0]) / 2
            center_lat = (left[1] + right[1]) / 2
            x, _y = proj.to_local(center_lon, center_lat)
            record = dict(gate_record)
            record['left_endpoint'] = {'lon': left[0], 'lat': left[1]}
            record['right_endpoint'] = {'lon': right[0], 'lat': right[1]}
            record['station_m'] = round(x, 3)
            record['source'] = 'kml_explicit'
            new_gates.append(record)
        else:
            new_gates.append(gate_record)
    result['gates'] = new_gates
    return result
