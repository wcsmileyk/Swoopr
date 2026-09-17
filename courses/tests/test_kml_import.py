"""
KML import tests using the real eastern Mile-Hi fixtures in
docs/course_planning/ -- these must import unchanged per design doc
section 6's acceptance tests A01/A02.
"""

import os

from django.test import SimpleTestCase

from courses.imports.kml import (
    KmlImportError,
    build_preview,
    derive_entry_setup,
    detect_discipline,
    parse_kml_bytes,
)
from courses.tests.test_geometry import EASTERN_GATE_CENTERS

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'docs', 'course_planning')


def _read_fixture(name):
    with open(os.path.join(FIXTURE_DIR, name), 'rb') as f:
        return f.read()


SPEED_KML = 'mile_hi_eastern_speed.kml'
DISTANCE_KML = 'mile_hi_eastern_distance.kml'


class ParseSpeedKmlTests(SimpleTestCase):
    def setUp(self):
        self.parsed = parse_kml_bytes(_read_fixture(SPEED_KML))

    def test_course_name_and_description_extracted(self):
        self.assertEqual(self.parsed.course_name, 'Mile-Hi Eastern Carved Speed')
        self.assertIn('129.645627', self.parsed.description)

    def test_five_gates_found_with_inside_and_outside(self):
        self.assertEqual(set(self.parsed.gates.keys()), {1, 2, 3, 4, 5})
        for n, gate in self.parsed.gates.items():
            self.assertIsNotNone(gate.inside, f'G{n} missing Inside')
            self.assertIsNotNone(gate.outside, f'G{n} missing Outside')

    def test_g1_and_g5_have_centers_too(self):
        self.assertIsNotNone(self.parsed.gates[1].center)
        self.assertIsNotNone(self.parsed.gates[5].center)

    def test_boundary_and_centerline_are_not_treated_as_gates(self):
        # Those lines have dozens of points but must not create extra gate
        # numbers or corrupt the 5 real gates.
        self.assertEqual(len(self.parsed.gates), 5)

    def test_no_reconciliation_warnings_for_clean_fixture(self):
        # Gate line endpoints should match their point placemarks exactly
        # in this fixture (they were generated together).
        self.assertEqual(self.parsed.warnings, [])

    def test_discipline_detected_as_speed(self):
        self.assertEqual(detect_discipline(self.parsed), 'speed')


class ParseDistanceKmlTests(SimpleTestCase):
    def setUp(self):
        self.parsed = parse_kml_bytes(_read_fixture(DISTANCE_KML))

    def test_course_name(self):
        self.assertEqual(self.parsed.course_name, 'Mile-Hi Eastern Drag Distance')

    def test_only_two_gates_g1_and_g5(self):
        # A02: no fabricated intermediate gates.
        self.assertEqual(set(self.parsed.gates.keys()), {1, 5})

    def test_discipline_detected_as_distance(self):
        self.assertEqual(detect_discipline(self.parsed), 'distance')

    def test_no_landing_cutoff_at_g5(self):
        # The file's boundary lines stop at 50m, but that must not appear
        # anywhere in the parsed gate data as a hard limit.
        self.assertNotIn('max_distance_m', self.parsed.__dict__)


class DeriveEntrySetupTests(SimpleTestCase):
    def test_speed_setup_matches_known_values(self):
        parsed = parse_kml_bytes(_read_fixture(SPEED_KML))
        setup, warnings = derive_entry_setup(parsed, carve_direction='left')
        self.assertAlmostEqual(setup.g1_lon, -105.165343056, places=6)
        self.assertAlmostEqual(setup.g1_lat, 40.161709722, places=6)
        self.assertAlmostEqual(setup.course_width_m, 10.014, delta=0.01)
        # Heading derived from G1->G5 should be close to the documented
        # 129.6456265 degrees true.
        self.assertAlmostEqual(setup.entry_heading_deg_true, 129.6456265, delta=0.01)
        self.assertEqual(warnings, [])

    def test_distance_setup_matches_speed_setup_shared_g1(self):
        # Both KMLs share the exact same physical G1 -- their derived
        # entry setups should agree closely even though heading is derived
        # from a different second gate (G5 at 50m vs G5 at 70m along the arc).
        speed_parsed = parse_kml_bytes(_read_fixture(SPEED_KML))
        distance_parsed = parse_kml_bytes(_read_fixture(DISTANCE_KML))
        speed_setup, _ = derive_entry_setup(speed_parsed, carve_direction='left')
        distance_setup, _ = derive_entry_setup(distance_parsed, carve_direction='left')
        self.assertAlmostEqual(speed_setup.g1_lon, distance_setup.g1_lon, places=6)
        self.assertAlmostEqual(speed_setup.g1_lat, distance_setup.g1_lat, places=6)

    def test_missing_g1_raises(self):
        parsed = parse_kml_bytes(_read_fixture(DISTANCE_KML))
        del parsed.gates[1]
        with self.assertRaises(KmlImportError):
            derive_entry_setup(parsed, carve_direction='left')


G1_SEED_KML = b'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
<name>Mile-Hi Eastern Seed</name>
<Placemark><name>G1 Center</name><Point><coordinates>-105.165343056,40.161709722,0</coordinates></Point></Placemark>
</Document></kml>'''


class G1SeedModeTests(SimpleTestCase):
    """A file with only a G1 marker and no gate line -- heading and width
    must be supplied explicitly (design doc section 6, 'G1 center Point')."""

    def setUp(self):
        self.parsed = parse_kml_bytes(G1_SEED_KML)

    def test_only_g1_parsed(self):
        self.assertEqual(set(self.parsed.gates.keys()), {1})
        self.assertIsNone(self.parsed.gates[1].inside)
        self.assertIsNotNone(self.parsed.gates[1].center)

    def test_discipline_not_auto_detected(self):
        self.assertIsNone(detect_discipline(self.parsed))

    def test_missing_heading_raises(self):
        with self.assertRaises(KmlImportError):
            derive_entry_setup(self.parsed, carve_direction='left')

    def test_explicit_heading_and_default_width(self):
        setup, warnings = derive_entry_setup(
            self.parsed, carve_direction='left', entry_heading_deg_true=129.6456265,
        )
        self.assertAlmostEqual(setup.g1_lon, -105.165343056, places=6)
        self.assertAlmostEqual(setup.course_width_m, 10.0)
        self.assertTrue(any('default' in w for w in warnings))

    def test_explicit_heading_and_width(self):
        setup, warnings = derive_entry_setup(
            self.parsed, carve_direction='left',
            entry_heading_deg_true=129.6456265, course_width_m=10.014,
        )
        self.assertAlmostEqual(setup.course_width_m, 10.014)
        self.assertEqual(warnings, [])

    def test_build_preview_requires_explicit_disciplines(self):
        with self.assertRaises(KmlImportError):
            build_preview(self.parsed, carve_direction='left', entry_heading_deg_true=129.6456265)

    def test_build_preview_generates_all_three_disciplines(self):
        preview = build_preview(
            self.parsed, carve_direction='left', entry_heading_deg_true=129.6456265,
            course_width_m=10.014, disciplines=['distance', 'speed', 'accuracy'],
        )
        self.assertEqual(set(preview['final_geometry'].keys()), {'distance', 'speed', 'accuracy'})
        # Nothing to substitute (no gate markers beyond G1's seed point) --
        # every gate should be purely generated.
        for geo in preview['final_geometry'].values():
            for gate in geo['gates']:
                self.assertEqual(gate['source'], 'generated_from_g1')
        # Speed geometry from this seed should match the real fixture's
        # own gates (same G1, same heading/width as the full Speed KML).
        speed_gates = preview['final_geometry']['speed']['gates']
        for gate in speed_gates:
            expected = EASTERN_GATE_CENTERS[gate['gate_number']]
            actual_lon = (gate['left_endpoint']['lon'] + gate['right_endpoint']['lon']) / 2
            actual_lat = (gate['left_endpoint']['lat'] + gate['right_endpoint']['lat']) / 2
            from courses.geometry import haversine_m
            self.assertLess(haversine_m(*expected, actual_lon, actual_lat), 0.05)


class BuildPreviewTests(SimpleTestCase):
    def test_speed_preview_has_low_deviation_from_template(self):
        parsed = parse_kml_bytes(_read_fixture(SPEED_KML))
        preview = build_preview(parsed, carve_direction='left')
        self.assertEqual(preview['detected_discipline'], 'speed')
        self.assertEqual(preview['disciplines'], ['speed'])
        for comparison in preview['template_comparisons']:
            self.assertLess(
                comparison['deviation_m'], 0.05,
                f"gate {comparison['gate_number']} deviates {comparison['deviation_m']}m from template"
            )

    def test_distance_preview_two_gates_no_fabrication(self):
        parsed = parse_kml_bytes(_read_fixture(DISTANCE_KML))
        preview = build_preview(parsed, carve_direction='left')
        self.assertEqual(preview['detected_discipline'], 'distance')
        gate_numbers = {g['gate_number'] for g in preview['generated_geometry']['distance']['gates']}
        self.assertEqual(gate_numbers, {1, 5})

    def test_final_geometry_preserves_imported_coordinates_exactly(self):
        # A01/A02: explicit geometry import must preserve uploaded
        # positions, not silently substitute canonical template coordinates.
        parsed = parse_kml_bytes(_read_fixture(SPEED_KML))
        preview = build_preview(parsed, carve_direction='left')
        g1_final = preview['final_geometry']['speed']['gates'][0]
        # Real KML G1 Inside/Outside (carve_direction='left' => inside=left)
        self.assertAlmostEqual(g1_final['left_endpoint']['lon'], -105.165305556, places=8)
        self.assertAlmostEqual(g1_final['left_endpoint']['lat'], 40.161744444, places=8)
        self.assertAlmostEqual(g1_final['right_endpoint']['lon'], -105.165380556, places=8)
        self.assertAlmostEqual(g1_final['right_endpoint']['lat'], 40.161675000, places=8)
        self.assertEqual(g1_final['source'], 'kml_explicit')

    def test_final_geometry_for_non_detected_discipline_is_generated(self):
        # Distance/Accuracy seeded from a Speed KML's G1 must not reuse the
        # Speed file's own G2-G4 as if they belonged to another discipline.
        parsed = parse_kml_bytes(_read_fixture(SPEED_KML))
        preview = build_preview(parsed, carve_direction='left', disciplines=['distance'])
        g5_final = preview['final_geometry']['distance']['gates'][1]
        self.assertEqual(g5_final['source'], 'generated_from_g1')
        self.assertEqual(g5_final['station_m'], 50.0)

    def test_explicit_discipline_overrides_detection(self):
        # A KML with only G1 present via a seed request could be asked to
        # generate a different discipline than what was auto-detected.
        parsed = parse_kml_bytes(_read_fixture(SPEED_KML))
        preview = build_preview(parsed, carve_direction='left', disciplines=['distance'])
        self.assertEqual(preview['disciplines'], ['distance'])
        self.assertIn('distance', preview['generated_geometry'])
        self.assertNotIn('speed', preview['generated_geometry'])


class KmlSecurityTests(SimpleTestCase):
    def test_rejects_doctype(self):
        malicious = b'''<?xml version="1.0"?>
<!DOCTYPE kml [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>&xxe;</name></Document></kml>'''
        with self.assertRaises(KmlImportError):
            parse_kml_bytes(malicious)

    def test_rejects_networklink(self):
        malicious = b'''<?xml version="1.0"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
<NetworkLink><Link><href>http://evil.example/payload.kml</href></Link></NetworkLink>
</Document></kml>'''
        with self.assertRaises(KmlImportError):
            parse_kml_bytes(malicious)

    def test_rejects_oversized_file(self):
        from courses.imports.kml import MAX_FILE_BYTES
        oversized = b'<kml>' + b' ' * (MAX_FILE_BYTES + 1) + b'</kml>'
        with self.assertRaises(KmlImportError):
            parse_kml_bytes(oversized)

    def test_rejects_out_of_range_coordinates(self):
        malicious = b'''<?xml version="1.0"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
<Placemark><name>G1 Inside</name><Point><coordinates>200,40,0</coordinates></Point></Placemark>
</Document></kml>'''
        with self.assertRaises(KmlImportError):
            parse_kml_bytes(malicious)

    def test_rejects_no_document_element(self):
        malicious = b'<?xml version="1.0"?><notkml></notkml>'
        with self.assertRaises(KmlImportError):
            parse_kml_bytes(malicious)
