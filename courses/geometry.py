"""
Pure geometry generation for competition courses.

No Django models or database access here -- everything takes and returns
plain floats/dicts so it can be unit tested without a database and reused
by both the generator (seed -> course) and the KML importer (validate
imported coordinates against the template).

Local coordinate convention (matches
docs/course_planning/Swoopr_Course_System_Requirements.md section 8):
  - G1 center is the origin.
  - +x is forward (down the course, in the direction of travel at G1).
  - +y is left (from the pilot's perspective flying forward).
  - Headings are true-north compass bearings in degrees (0 = north,
    clockwise positive), describing the pilot's direction of travel at G1 --
    not the bearing across the gate.

Projection: rather than a flat degrees-to-meters approximation, points are
projected through a local azimuthal-equidistant projection centered on G1
(already available via GDAL/PROJ, which GeoDjango requires anyway), so
true headings and meters from G1 are exact by construction rather than
approximated. See LocalProjection for why this was chosen over a fixed
UTM zone.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from django.contrib.gis.gdal import SpatialReference, CoordTransform
from django.contrib.gis.geos import Point

_WGS84 = SpatialReference(4326)

GEOMETRY_ALGORITHM_VERSION = 'courses-geometry-v1'

SPEED_ARC_ANGLE_DEG = 75.0
SPEED_ARC_LENGTH_M = 70.0
SPEED_GATE_COUNT = 5
SPEED_BOUNDARY_SAMPLE_COUNT = 48
DEFAULT_COURSE_WIDTH_M = 10.0

DISTANCE_LENGTH_M = 50.0

# Zone Accuracy longitudinal layout, meters from G1 (see design doc section 8,
# visually verified against Annex F.3 and the supplied layout screenshot).
ACCURACY_WATER_GATE_STATIONS_M = [0.0, 12.0, 24.0, 36.0]
ACCURACY_WATERLINE_M = 44.0
ACCURACY_ZONE_WIDTH_Y_M = 5.0  # zones extend +/-5m laterally except Z7/Z8/CZ
ACCURACY_WATER_GATE_POINTS = [21, 5, 8, 16]  # G1..G4, per Annex F.3 diagram

# (zone_id, start_m, end_m, points) for the wide zones between the waterline
# and the split Z7/Z8/CZ block, plus the zones beyond it.
ACCURACY_WIDE_ZONES = [
    ('Z1', 44.0, 50.0, 3),
    ('Z2', 50.0, 56.0, 11),
    ('Z3', 56.0, 61.0, 19),
    ('Z4', 61.0, 65.0, 27),
    ('Z5', 65.0, 68.0, 34),
    ('Z6', 68.0, 70.0, 41),
    ('Z9', 72.0, 74.0, 25),
    ('Z10', 74.0, 78.0, 5),
]

# The split block from 70-72m: each entry is (zone_id, y_min, y_max, points).
# Symmetric about the centerline; listed once per side.
ACCURACY_SPLIT_BLOCK_START_M = 70.0
ACCURACY_SPLIT_BLOCK_END_M = 72.0
ACCURACY_SPLIT_LATERAL_BANDS = [
    ('Z7', 1.5, 5.0, 46),
    ('Z8', 0.5, 1.5, 48),
    ('CZ', 0.0, 0.5, 50),
]


class GeometryError(ValueError):
    """Raised for invalid or physically-impossible course inputs."""


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in meters between two WGS84 points. Good
    enough for tolerance checks (cm-level near a course); not the
    projection used for actual geometry generation -- see LocalProjection.
    """
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


@dataclass(frozen=True)
class LocalProjection:
    """A local azimuthal-equidistant projection centered exactly on
    (origin_lon, origin_lat).

    Chosen over a fixed UTM zone because AEQD is exact from its center
    point outward: true north and true distance FROM THE ORIGIN are
    preserved by construction, with no grid-convergence correction needed.
    A UTM zone's grid north only equals true north on its central
    meridian -- everywhere else (including here) it's off by a small but
    real convergence angle that, left uncorrected, put the far end of a
    70m speed course several centimeters off from the true-heading
    construction. Course markers are within ~100m of the origin, well
    inside AEQD's low-distortion range.
    """
    origin_lon: float
    origin_lat: float

    def _transform(self) -> SpatialReference:
        return SpatialReference(
            f'+proj=aeqd +lat_0={self.origin_lat} +lon_0={self.origin_lon} '
            f'+datum=WGS84 +units=m +no_defs'
        )

    @classmethod
    def at(cls, lon: float, lat: float) -> 'LocalProjection':
        return cls(origin_lon=lon, origin_lat=lat)

    def to_local(self, lon: float, lat: float) -> Tuple[float, float]:
        """WGS84 (lon, lat) -> (east_m, north_m) relative to the origin."""
        aeqd = self._transform()
        p = Point(lon, lat, srid=4326)
        p.transform(CoordTransform(_WGS84, aeqd))
        return p.x, p.y

    def to_lonlat(self, east_m: float, north_m: float) -> Tuple[float, float]:
        """(east_m, north_m) relative to the origin -> WGS84 (lon, lat)."""
        aeqd = self._transform()
        p = Point(east_m, north_m)
        p.transform(CoordTransform(aeqd, _WGS84))
        return p.x, p.y


def forward_left_to_east_north(x_forward: float, y_left: float, heading_deg: float) -> Tuple[float, float]:
    """Rotate local (forward, left) meters into (east, north) meters for a
    true-north compass heading in degrees."""
    h = math.radians(heading_deg)
    east = x_forward * math.sin(h) - y_left * math.cos(h)
    north = x_forward * math.cos(h) + y_left * math.sin(h)
    return east, north


def east_north_to_forward_left(east: float, north: float, heading_deg: float) -> Tuple[float, float]:
    """Inverse of forward_left_to_east_north (the rotation is orthonormal,
    so its inverse is its transpose)."""
    h = math.radians(heading_deg)
    x_forward = east * math.sin(h) + north * math.cos(h)
    y_left = -east * math.cos(h) + north * math.sin(h)
    return x_forward, y_left


@dataclass(frozen=True)
class EntrySetup:
    """The minimal inputs needed to generate any/all discipline geometry."""
    g1_lon: float
    g1_lat: float
    entry_heading_deg_true: float
    carve_direction: str  # 'left' or 'right' (only meaningful for Speed)
    course_width_m: float = DEFAULT_COURSE_WIDTH_M

    def __post_init__(self):
        if self.carve_direction not in ('left', 'right'):
            raise GeometryError(f"carve_direction must be 'left' or 'right', got {self.carve_direction!r}")
        if self.course_width_m <= 0:
            raise GeometryError('course_width_m must be positive')

    @property
    def projection(self) -> LocalProjection:
        return LocalProjection.at(self.g1_lon, self.g1_lat)

    def place(self, x_forward: float, y_left: float) -> Tuple[float, float]:
        """Local (forward, left) meters from G1 -> WGS84 (lon, lat)."""
        east, north = forward_left_to_east_north(x_forward, y_left, self.entry_heading_deg_true)
        return self.projection.to_lonlat(east, north)

    def to_forward_left(self, lon: float, lat: float) -> Tuple[float, float]:
        """WGS84 (lon, lat) -> local (forward, left) meters from G1. Inverse
        of place(); this is what "how far down the course" actually means
        -- not the raw east/north offset, which points wherever the
        compass happens to say and is only equal to the forward distance
        when the entry heading is due north."""
        east, north = self.projection.to_local(lon, lat)
        return east_north_to_forward_left(east, north, self.entry_heading_deg_true)

    def gate_endpoints(self, x_forward: float, y_left_center: float = 0.0,
                        width: float = None) -> Dict[str, Tuple[float, float]]:
        """Left/right endpoint lon/lat for a gate centered on the local
        centerline point (x_forward, y_left_center), perpendicular to the
        direction of travel there (straight courses: perpendicular to
        entry_heading_deg_true)."""
        w = self.course_width_m if width is None else width
        left = self.place(x_forward, y_left_center + w / 2)
        right = self.place(x_forward, y_left_center - w / 2)
        return {'left': left, 'right': right}


def _polyline(setup: EntrySetup, local_points: List[Tuple[float, float]]) -> List[Dict]:
    return [{'lon': lon, 'lat': lat} for lon, lat in (setup.place(x, y) for x, y in local_points)]


def _straight_boundary(setup: EntrySetup, x_start: float, x_end: float, half_width: float) -> Dict:
    return {
        'left': _polyline(setup, [(x_start, half_width), (x_end, half_width)]),
        'right': _polyline(setup, [(x_start, -half_width), (x_end, -half_width)]),
        'centerline': _polyline(setup, [(x_start, 0.0), (x_end, 0.0)]),
    }


def _gate_record(gate_id: str, gate_number: int, station_m: float,
                  left_lonlat: Tuple[float, float], right_lonlat: Tuple[float, float],
                  source: str) -> Dict:
    return {
        'id': gate_id,
        'gate_number': gate_number,
        'station_m': round(station_m, 3),
        'left_endpoint': {'lon': left_lonlat[0], 'lat': left_lonlat[1]},
        'right_endpoint': {'lon': right_lonlat[0], 'lat': right_lonlat[1]},
        'measurement_reference': 'marker_center',
        'source': source,
    }


def generate_distance_geometry(setup: EntrySetup) -> Dict:
    """G1 at x=0, G5 at x=50, straight 10m-wide corridor."""
    g1 = setup.gate_endpoints(0.0)
    g5 = setup.gate_endpoints(DISTANCE_LENGTH_M)
    return {
        'discipline': 'distance',
        'algorithm_version': GEOMETRY_ALGORITHM_VERSION,
        'entry_heading_deg_true': setup.entry_heading_deg_true,
        'course_width_m': setup.course_width_m,
        'length_m': DISTANCE_LENGTH_M,
        'gates': [
            _gate_record('distance:G1', 1, 0.0, g1['left'], g1['right'], 'generated_from_g1'),
            _gate_record('distance:G5', 5, DISTANCE_LENGTH_M, g5['left'], g5['right'], 'generated_from_g1'),
        ],
        'boundary': _straight_boundary(setup, 0.0, DISTANCE_LENGTH_M, setup.course_width_m / 2),
    }


def _speed_arc_offset_point(radius: float, phi: float, s: float, lateral_offset: float) -> Tuple[float, float]:
    """Local (x_forward, y_left) of a point on the 75-degree arc at angle
    phi, offset laterally by lateral_offset (positive = toward the pilot's
    left) from the centerline."""
    center_x = radius * math.sin(phi)
    center_y = s * radius * (1 - math.cos(phi))
    # local tangent direction is (cos(phi), s*sin(phi)); left-of-travel
    # normal is (-s*sin(phi), cos(phi))
    left_x = -s * math.sin(phi)
    left_y = math.cos(phi)
    return center_x + lateral_offset * left_x, center_y + lateral_offset * left_y


def generate_speed_geometry(setup: EntrySetup) -> Dict:
    """Curved 70m course through a 75-degree arc, 5 gates, per the canonical
    formula in the design doc section 8."""
    theta_total = math.radians(SPEED_ARC_ANGLE_DEG)
    s = 1.0 if setup.carve_direction == 'left' else -1.0
    radius = SPEED_ARC_LENGTH_M / theta_total
    half_w = setup.course_width_m / 2

    gates = []
    for i in range(SPEED_GATE_COUNT):
        gate_number = i + 1
        phi = theta_total * i / (SPEED_GATE_COUNT - 1)
        station_m = radius * phi

        left_pt = setup.place(*_speed_arc_offset_point(radius, phi, s, half_w))
        right_pt = setup.place(*_speed_arc_offset_point(radius, phi, s, -half_w))

        gates.append(_gate_record(f'speed:G{gate_number}', gate_number, station_m, left_pt, right_pt, 'generated_from_g1'))

    sample_phis = [theta_total * i / SPEED_BOUNDARY_SAMPLE_COUNT for i in range(SPEED_BOUNDARY_SAMPLE_COUNT + 1)]
    boundary = {
        'left': _polyline(setup, [_speed_arc_offset_point(radius, phi, s, half_w) for phi in sample_phis]),
        'right': _polyline(setup, [_speed_arc_offset_point(radius, phi, s, -half_w) for phi in sample_phis]),
        'centerline': _polyline(setup, [_speed_arc_offset_point(radius, phi, s, 0.0) for phi in sample_phis]),
    }

    return {
        'discipline': 'speed',
        'algorithm_version': GEOMETRY_ALGORITHM_VERSION,
        'entry_heading_deg_true': setup.entry_heading_deg_true,
        'course_width_m': setup.course_width_m,
        'carve_direction': setup.carve_direction,
        'arc_angle_deg': SPEED_ARC_ANGLE_DEG,
        'arc_radius_m': round(radius, 3),
        'arc_length_m': SPEED_ARC_LENGTH_M,
        'gates': gates,
        'boundary': boundary,
    }


def _accuracy_zone_polygon(setup: EntrySetup, start_m: float, end_m: float,
                            y_min: float, y_max: float) -> List[Dict]:
    corners_local = [
        (start_m, y_max), (end_m, y_max), (end_m, y_min), (start_m, y_min),
    ]
    return [{'lon': lon, 'lat': lat} for lon, lat in (setup.place(x, y) for x, y in corners_local)]


def generate_accuracy_geometry(setup: EntrySetup) -> Dict:
    """Straight course: 4 water gates + Z1-Z10/CZ landing zones."""
    gates = []
    for i, station_m in enumerate(ACCURACY_WATER_GATE_STATIONS_M):
        gate_number = i + 1
        pts = setup.gate_endpoints(station_m)
        record = _gate_record(f'accuracy:G{gate_number}', gate_number, station_m,
                               pts['left'], pts['right'], 'generated_from_g1')
        record['points'] = ACCURACY_WATER_GATE_POINTS[i]
        gates.append(record)

    zones = []
    for zone_id, start_m, end_m, points in ACCURACY_WIDE_ZONES:
        zones.append({
            'id': f'accuracy:{zone_id}',
            'points': points,
            'longitudinal_extent_m': [start_m, end_m],
            'lateral_extent_m': [-ACCURACY_ZONE_WIDTH_Y_M, ACCURACY_ZONE_WIDTH_Y_M],
            'polygon': [_accuracy_zone_polygon(setup, start_m, end_m, -ACCURACY_ZONE_WIDTH_Y_M, ACCURACY_ZONE_WIDTH_Y_M)],
        })

    # Split block (Z7/Z8/CZ): each band appears on both the left and right
    # side of the centerline, forming a disconnected (MultiPolygon-style)
    # zone for Z7 and Z8.
    for zone_id, y_min, y_max, points in ACCURACY_SPLIT_LATERAL_BANDS:
        polygons = [
            _accuracy_zone_polygon(setup, ACCURACY_SPLIT_BLOCK_START_M, ACCURACY_SPLIT_BLOCK_END_M, y_min, y_max),
        ]
        if y_min > 0:
            # mirror to the right (negative y) side, except CZ which already
            # straddles the centerline (y_min == 0)
            polygons.append(
                _accuracy_zone_polygon(setup, ACCURACY_SPLIT_BLOCK_START_M, ACCURACY_SPLIT_BLOCK_END_M, -y_max, -y_min)
            )
        zones.append({
            'id': f'accuracy:{zone_id}',
            'points': points,
            'longitudinal_extent_m': [ACCURACY_SPLIT_BLOCK_START_M, ACCURACY_SPLIT_BLOCK_END_M],
            'lateral_extent_m': [y_min, y_max],
            'polygon': polygons,
        })

    cz_center_m = (ACCURACY_SPLIT_BLOCK_START_M + ACCURACY_SPLIT_BLOCK_END_M) / 2
    course_end_m = ACCURACY_WIDE_ZONES[-1][2]  # end of Z10

    return {
        'discipline': 'accuracy',
        'algorithm_version': GEOMETRY_ALGORITHM_VERSION,
        'entry_heading_deg_true': setup.entry_heading_deg_true,
        'course_width_m': setup.course_width_m,
        'waterline_m': ACCURACY_WATERLINE_M,
        'center_zone_station_m': cz_center_m,
        'gates': gates,
        'zones': zones,
        # Boundary matches the zones' own +/-5m lateral extent (not
        # course_width_m/2, which only governs the water-gate spacing) so
        # the rendered rails actually bound the landing-zone polygons.
        'boundary': _straight_boundary(setup, 0.0, course_end_m, ACCURACY_ZONE_WIDTH_Y_M),
    }


DISCIPLINE_GENERATORS = {
    'distance': generate_distance_geometry,
    'speed': generate_speed_geometry,
    'accuracy': generate_accuracy_geometry,
}


def generate_geometry(setup: EntrySetup, disciplines: List[str]) -> Dict[str, Dict]:
    """Generate one or more discipline geometries sharing the same entry
    setup. Raises GeometryError for an unknown discipline name."""
    result = {}
    for discipline in disciplines:
        generator = DISCIPLINE_GENERATORS.get(discipline)
        if generator is None:
            raise GeometryError(f'Unknown discipline: {discipline!r}')
        result[discipline] = generator(setup)
    return result
