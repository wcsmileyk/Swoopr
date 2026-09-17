"""
Geometry tests, validated against the real mile_hi_eastern_speed.kml /
mile_hi_eastern_distance.kml fixtures in docs/course_planning/ wherever
possible -- not just internal self-consistency.

Tolerances: 1cm for pure math self-consistency checks (matches the design
doc's suggested regression tolerance for constructed local-coordinate
fixtures); 5cm when comparing against the existing KML's rounded
centerline-radius (53.48 vs the exact 70/radians(75)) construction, per
the design doc's stated computational compatibility tolerance for that file.
"""

from django.test import SimpleTestCase

from courses.geometry import (
    EntrySetup,
    GeometryError,
    LocalProjection,
    generate_accuracy_geometry,
    generate_distance_geometry,
    generate_speed_geometry,
    haversine_m as _haversine_m,
)

# From docs/course_planning/mile_hi_eastern_speed.kml
EASTERN_G1_CENTER = (-105.165343056, 40.161709722)
EASTERN_ENTRY_HEADING = 129.6456265
EASTERN_G1_WIDTH_M = 10.014

EASTERN_GATE_CENTERS = {
    1: (-105.165343056, 40.161709722),
    2: (-105.165166423, 40.161630625),
    3: (-105.164966025, 40.161599286),
    4: (-105.164763134, 40.161619032),
    5: (-105.164579282, 40.161687768),
}


class LocalProjectionRoundTripTests(SimpleTestCase):
    def test_origin_maps_to_zero(self):
        proj = LocalProjection.at(*EASTERN_G1_CENTER)
        x, y = proj.to_local(*EASTERN_G1_CENTER)
        self.assertAlmostEqual(x, 0.0, delta=0.001)
        self.assertAlmostEqual(y, 0.0, delta=0.001)

    def test_round_trip_lonlat_local_lonlat(self):
        proj = LocalProjection.at(*EASTERN_G1_CENTER)
        lon, lat = proj.to_lonlat(37.0, -21.0)
        x, y = proj.to_local(lon, lat)
        self.assertAlmostEqual(x, 37.0, delta=0.001)
        self.assertAlmostEqual(y, -21.0, delta=0.001)

    def test_known_bearing_and_distance(self):
        # 100m due north of the origin should land ~100m away at ~0deg true.
        # Tolerance is loose (not cm-level) because _haversine_m assumes a
        # sphere while the AEQD projection correctly uses the WGS84
        # ellipsoid -- the two models diverge by ~0.1-0.15% at this
        # latitude for a distance-from-origin check. This is an artifact of
        # the spherical verification formula, not the ellipsoidal
        # projection being tested; see test_matches_real_eastern_kml_gate_centers
        # for a tight point-to-point check where this doesn't apply.
        proj = LocalProjection.at(*EASTERN_G1_CENTER)
        lon, lat = proj.to_lonlat(0.0, 100.0)
        dist = _haversine_m(EASTERN_G1_CENTER[0], EASTERN_G1_CENTER[1], lon, lat)
        self.assertAlmostEqual(dist, 100.0, delta=0.2)


class ForwardLeftRoundTripTests(SimpleTestCase):
    def setUp(self):
        self.setup = EntrySetup(
            g1_lon=EASTERN_G1_CENTER[0], g1_lat=EASTERN_G1_CENTER[1],
            entry_heading_deg_true=EASTERN_ENTRY_HEADING,
            carve_direction='left', course_width_m=10.0,
        )

    def test_place_then_to_forward_left_round_trips(self):
        lon, lat = self.setup.place(50.0, -3.0)
        x, y = self.setup.to_forward_left(lon, lat)
        self.assertAlmostEqual(x, 50.0, delta=0.01)
        self.assertAlmostEqual(y, -3.0, delta=0.01)

    def test_g1_center_is_the_origin(self):
        x, y = self.setup.to_forward_left(EASTERN_G1_CENTER[0], EASTERN_G1_CENTER[1])
        self.assertAlmostEqual(x, 0.0, delta=0.01)
        self.assertAlmostEqual(y, 0.0, delta=0.01)

    def test_distance_g5_of_real_kml_is_50m_forward_not_raw_offset(self):
        # Regression guard: this must be the ROTATED forward distance, not
        # the raw east/north offset from G1 (those only coincide when
        # heading is due north -- here it's ~130 degrees, so a raw-offset
        # bug would give a very different, wrong number). Uses the real
        # eastern Distance KML's G5 center (straight course, so forward
        # distance and physical separation are the same 50m -- unlike a
        # curved Speed gate, where they deliberately differ).
        g5_lon, g5_lat = -105.164891132, 40.161422416  # real eastern Distance KML G5 center
        x, y = self.setup.to_forward_left(g5_lon, g5_lat)
        self.assertAlmostEqual(x, 50.0, delta=0.1)
        self.assertAlmostEqual(y, 0.0, delta=0.1)


class EntrySetupTests(SimpleTestCase):
    def test_rejects_invalid_carve_direction(self):
        with self.assertRaises(GeometryError):
            EntrySetup(g1_lon=0, g1_lat=0, entry_heading_deg_true=90, carve_direction='up')

    def test_rejects_non_positive_width(self):
        with self.assertRaises(GeometryError):
            EntrySetup(g1_lon=0, g1_lat=0, entry_heading_deg_true=90, carve_direction='left', course_width_m=0)

    def test_g1_gate_width_matches_input(self):
        setup = EntrySetup(g1_lon=EASTERN_G1_CENTER[0],
                            g1_lat=EASTERN_G1_CENTER[1],
                            entry_heading_deg_true=EASTERN_ENTRY_HEADING,
                            carve_direction='left', course_width_m=10.0)
        pts = setup.gate_endpoints(0.0)
        dist = _haversine_m(pts['left'][0], pts['left'][1], pts['right'][0], pts['right'][1])
        self.assertAlmostEqual(dist, 10.0, delta=0.01)


class DistanceGeometryTests(SimpleTestCase):
    def setUp(self):
        self.setup = EntrySetup(
            g1_lon=EASTERN_G1_CENTER[0], g1_lat=EASTERN_G1_CENTER[1],
            entry_heading_deg_true=EASTERN_ENTRY_HEADING,
            carve_direction='left', course_width_m=10.0,
        )

    def test_two_gates_at_0_and_50m(self):
        geo = generate_distance_geometry(self.setup)
        self.assertEqual(len(geo['gates']), 2)
        self.assertEqual(geo['gates'][0]['station_m'], 0.0)
        self.assertEqual(geo['gates'][1]['station_m'], 50.0)

    def test_g5_is_50m_from_g1_along_heading(self):
        # delta is loose for the same sphere-vs-ellipsoid reason as
        # test_known_bearing_and_distance above -- this checks distance
        # from the origin (G1), not point-to-point proximity.
        geo = generate_distance_geometry(self.setup)
        g1 = geo['gates'][0]
        g5 = geo['gates'][1]
        g1_center = ((g1['left_endpoint']['lon'] + g1['right_endpoint']['lon']) / 2,
                     (g1['left_endpoint']['lat'] + g1['right_endpoint']['lat']) / 2)
        g5_center = ((g5['left_endpoint']['lon'] + g5['right_endpoint']['lon']) / 2,
                     (g5['left_endpoint']['lat'] + g5['right_endpoint']['lat']) / 2)
        dist = _haversine_m(*g1_center, *g5_center)
        self.assertAlmostEqual(dist, 50.0, delta=0.1)

    def test_no_landing_cutoff_encoded_in_length(self):
        # A50m KML file must not imply distances beyond 50m are invalid --
        # generate_distance_geometry only describes the two required gates.
        geo = generate_distance_geometry(self.setup)
        self.assertEqual(geo['length_m'], 50.0)
        self.assertNotIn('max_distance_m', geo)

    def test_boundary_lines_span_full_length_at_half_width(self):
        geo = generate_distance_geometry(self.setup)
        boundary = geo['boundary']
        for side in ('left', 'right', 'centerline'):
            self.assertEqual(len(boundary[side]), 2)
        left_start, left_end = boundary['left']
        dist = _haversine_m(left_start['lon'], left_start['lat'], left_end['lon'], left_end['lat'])
        self.assertAlmostEqual(dist, 50.0, delta=0.1)
        right_start = boundary['right'][0]
        width = _haversine_m(left_start['lon'], left_start['lat'], right_start['lon'], right_start['lat'])
        self.assertAlmostEqual(width, 10.0, delta=0.01)


class SpeedGeometryTests(SimpleTestCase):
    def setUp(self):
        self.setup = EntrySetup(
            g1_lon=EASTERN_G1_CENTER[0], g1_lat=EASTERN_G1_CENTER[1],
            entry_heading_deg_true=EASTERN_ENTRY_HEADING,
            carve_direction='left', course_width_m=EASTERN_G1_WIDTH_M,
        )
        self.geo = generate_speed_geometry(self.setup)

    def test_five_gates_at_correct_stations(self):
        stations = [g['station_m'] for g in self.geo['gates']]
        self.assertEqual(len(stations), 5)
        self.assertAlmostEqual(stations[0], 0.0, delta=0.001)
        self.assertAlmostEqual(stations[-1], 70.0, delta=0.01)
        # Equal 17.5m spacing (70m over 4 intervals)
        deltas = [stations[i + 1] - stations[i] for i in range(4)]
        for d in deltas:
            self.assertAlmostEqual(d, 17.5, delta=0.01)

    def test_matches_real_eastern_kml_gate_centers(self):
        for gate in self.geo['gates']:
            expected = EASTERN_GATE_CENTERS[gate['gate_number']]
            actual_lon = (gate['left_endpoint']['lon'] + gate['right_endpoint']['lon']) / 2
            actual_lat = (gate['left_endpoint']['lat'] + gate['right_endpoint']['lat']) / 2
            dist = _haversine_m(expected[0], expected[1], actual_lon, actual_lat)
            self.assertLess(
                dist, 0.05,
                f"gate {gate['gate_number']} is {dist:.3f}m from the real KML center "
                f"(tolerance 5cm for the KML's rounded-radius construction)"
            )

    def test_boundary_has_sampled_arc_points_at_correct_endpoints(self):
        boundary = self.geo['boundary']
        self.assertGreater(len(boundary['left']), 10)
        self.assertEqual(len(boundary['left']), len(boundary['right']))
        # First boundary sample should coincide with G1, last with G5.
        g1, g5 = self.geo['gates'][0], self.geo['gates'][-1]
        first_left, last_left = boundary['left'][0], boundary['left'][-1]
        self.assertAlmostEqual(first_left['lon'], g1['left_endpoint']['lon'], places=6)
        self.assertAlmostEqual(first_left['lat'], g1['left_endpoint']['lat'], places=6)
        self.assertAlmostEqual(last_left['lon'], g5['left_endpoint']['lon'], places=6)
        self.assertAlmostEqual(last_left['lat'], g5['left_endpoint']['lat'], places=6)

    def test_right_carve_mirrors_left_carve(self):
        right_setup = EntrySetup(
            g1_lon=EASTERN_G1_CENTER[0], g1_lat=EASTERN_G1_CENTER[1],
            entry_heading_deg_true=EASTERN_ENTRY_HEADING,
            carve_direction='right', course_width_m=EASTERN_G1_WIDTH_M,
        )
        right_geo = generate_speed_geometry(right_setup)
        # G1 should be identical regardless of carve direction (it's the
        # shared entry gate); G5 should differ (opposite side of the arc).
        left_g1 = self.geo['gates'][0]
        right_g1 = right_geo['gates'][0]
        self.assertAlmostEqual(left_g1['left_endpoint']['lon'], right_g1['left_endpoint']['lon'], places=6)
        left_g5 = self.geo['gates'][-1]
        right_g5 = right_geo['gates'][-1]
        self.assertNotAlmostEqual(left_g5['left_endpoint']['lat'], right_g5['left_endpoint']['lat'], places=4)


class AccuracyGeometryTests(SimpleTestCase):
    def setUp(self):
        self.setup = EntrySetup(
            g1_lon=EASTERN_G1_CENTER[0], g1_lat=EASTERN_G1_CENTER[1],
            entry_heading_deg_true=EASTERN_ENTRY_HEADING,
            carve_direction='left', course_width_m=10.0,
        )
        self.geo = generate_accuracy_geometry(self.setup)

    def test_four_water_gates_with_correct_points(self):
        self.assertEqual(len(self.geo['gates']), 4)
        self.assertEqual([g['points'] for g in self.geo['gates']], [21, 5, 8, 16])
        self.assertEqual([g['station_m'] for g in self.geo['gates']], [0.0, 12.0, 24.0, 36.0])

    def test_center_zone_is_71m_not_67m(self):
        # Regression guard for the corrected CZ position (see design doc
        # section 8's "Correction to earlier messages").
        self.assertAlmostEqual(self.geo['center_zone_station_m'], 71.0, delta=0.001)

    def test_zone_point_values_match_diagram(self):
        expected_points = {
            'accuracy:Z1': 3, 'accuracy:Z2': 11, 'accuracy:Z3': 19, 'accuracy:Z4': 27,
            'accuracy:Z5': 34, 'accuracy:Z6': 41, 'accuracy:Z7': 46, 'accuracy:Z8': 48,
            'accuracy:CZ': 50, 'accuracy:Z9': 25, 'accuracy:Z10': 5,
        }
        actual_points = {z['id']: z['points'] for z in self.geo['zones']}
        self.assertEqual(actual_points, expected_points)

    def test_all_ten_zones_plus_center_zone_present(self):
        zone_ids = {z['id'] for z in self.geo['zones']}
        expected_ids = {f'accuracy:Z{i}' for i in range(1, 11)} | {'accuracy:CZ'}
        self.assertEqual(zone_ids, expected_ids)

    def test_boundary_spans_full_78m_at_5m_half_width(self):
        boundary = self.geo['boundary']
        left_start, left_end = boundary['left']
        dist = _haversine_m(left_start['lon'], left_start['lat'], left_end['lon'], left_end['lat'])
        self.assertAlmostEqual(dist, 78.0, delta=0.1)
        right_start = boundary['right'][0]
        width = _haversine_m(left_start['lon'], left_start['lat'], right_start['lon'], right_start['lat'])
        self.assertAlmostEqual(width, 10.0, delta=0.01)  # +/-5m per side

    def test_split_block_zones_have_two_polygons(self):
        by_id = {z['id']: z for z in self.geo['zones']}
        # Z7 and Z8 are disconnected (left and right of centerline)
        self.assertEqual(len(by_id['accuracy:Z7']['polygon']), 2)
        self.assertEqual(len(by_id['accuracy:Z8']['polygon']), 2)
        # CZ straddles the centerline -- a single polygon
        self.assertEqual(len(by_id['accuracy:CZ']['polygon']), 1)
