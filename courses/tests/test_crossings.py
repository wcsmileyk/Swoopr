"""
Crossing-detection tests. These directly correspond to the design doc's
acceptance criteria A12-A15 for the crossing engine: interpolated center
crossing, outside-span crossing, reverse crossing, near-miss/far-away
tracks (no substituted closest-approach), exact-on-line points, and
repeated multi-segment crossings.
"""

from datetime import datetime, timedelta, timezone

from django.test import SimpleTestCase

from courses.analysis.crossings import GateFrame, analyze_course_crossings, analyze_gate_crossing
from courses.geometry import EntrySetup, generate_distance_geometry, generate_speed_geometry

BASE_TIME = datetime(2026, 9, 5, 18, 0, 0, tzinfo=timezone.utc)


def _setup():
    return EntrySetup(
        g1_lon=-105.165343056, g1_lat=40.161709722,
        entry_heading_deg_true=129.6456265,
        carve_direction='left', course_width_m=10.0,
    )


def _distance_g1_gate():
    return generate_distance_geometry(_setup())['gates'][0]


def _point_relative_to_gate(gate, forward_m, tangential_m, t, **extra):
    frame = GateFrame.from_gate(gate)
    east = forward_m * frame.forward[0] + tangential_m * frame.tangent[0]
    north = forward_m * frame.forward[1] + tangential_m * frame.tangent[1]
    lon, lat = frame.projection.to_lonlat(east, north)
    point = {'timestamp': t, 'lon': lon, 'lat': lat}
    point.update(extra)
    return point


class GateFrameTests(SimpleTestCase):
    def test_half_width_matches_gate_width(self):
        gate = _distance_g1_gate()
        frame = GateFrame.from_gate(gate)
        self.assertAlmostEqual(frame.half_width_m, 5.0, delta=0.01)

    def test_local_xy_of_gate_center_is_origin(self):
        gate = _distance_g1_gate()
        frame = GateFrame.from_gate(gate)
        center_lon = (gate['left_endpoint']['lon'] + gate['right_endpoint']['lon']) / 2
        center_lat = (gate['left_endpoint']['lat'] + gate['right_endpoint']['lat']) / 2
        x, y = frame.local_xy(center_lon, center_lat)
        self.assertAlmostEqual(x, 0.0, delta=0.001)
        self.assertAlmostEqual(y, 0.0, delta=0.001)

    def test_rejects_zero_length_gate(self):
        gate = _distance_g1_gate()
        gate = dict(gate, right_endpoint=dict(gate['left_endpoint']))
        with self.assertRaises(ValueError):
            GateFrame.from_gate(gate)


class CenterCrossingTests(SimpleTestCase):
    """A12: synthetic center crossing halfway between samples."""

    def test_interpolated_center_crossing(self):
        gate = _distance_g1_gate()
        points = [
            _point_relative_to_gate(gate, -5.0, 0.0, BASE_TIME, ground_speed=30.0, heading=130.0,
                                     altitude_agl=20.0, v_acc=0.8),
            _point_relative_to_gate(gate, 5.0, 0.0, BASE_TIME + timedelta(seconds=0.25), ground_speed=30.0, heading=130.0,
                                     altitude_agl=18.0, v_acc=1.0),
        ]
        result = analyze_gate_crossing(points, gate)
        self.assertEqual(result.status, 'crossed')
        self.assertTrue(result.within_span)
        self.assertAlmostEqual(result.alpha, 0.5, places=3)
        self.assertAlmostEqual(result.lateral_offset_m, 0.0, delta=0.01)
        self.assertEqual(result.crossing_time, BASE_TIME + timedelta(seconds=0.125))
        self.assertAlmostEqual(result.speed_mps, 30.0, delta=0.01)
        self.assertAlmostEqual(result.altitude_agl, 19.0, delta=0.01)
        self.assertAlmostEqual(result.vertical_accuracy_m, 0.9, delta=0.01)
        self.assertEqual(result.confidence, 'ok')


class OutsideSpanTests(SimpleTestCase):
    """A13: outside span."""

    def test_crossing_outside_finite_gate_span(self):
        gate = _distance_g1_gate()
        points = [
            _point_relative_to_gate(gate, -5.0, 8.0, BASE_TIME),
            _point_relative_to_gate(gate, 5.0, 8.0, BASE_TIME + timedelta(seconds=1)),
        ]
        result = analyze_gate_crossing(points, gate)
        self.assertEqual(result.status, 'crossed')
        self.assertFalse(result.within_span)
        self.assertTrue(any('out-of-span' in w for w in result.warnings))


class ReverseCrossingTests(SimpleTestCase):
    """A13: reverse crossing must not register as an entry."""

    def test_reverse_only_crossing_not_treated_as_entry(self):
        gate = _distance_g1_gate()
        points = [
            _point_relative_to_gate(gate, 5.0, 0.0, BASE_TIME),
            _point_relative_to_gate(gate, -5.0, 0.0, BASE_TIME + timedelta(seconds=1)),
        ]
        result = analyze_gate_crossing(points, gate)
        self.assertEqual(result.status, 'not_crossed')
        self.assertEqual(result.reverse_crossing_indices, [1])
        self.assertIsNone(result.crossing_time)


class NoCrossingDiagnosticTests(SimpleTestCase):
    """A13: near miss and far-away tracks never substitute a
    closest-approach for a real crossing."""

    def test_near_miss_reports_closest_approach_but_not_crossed(self):
        gate = _distance_g1_gate()
        points = [
            _point_relative_to_gate(gate, -5.0, 0.0, BASE_TIME),
            _point_relative_to_gate(gate, -0.5, 0.0, BASE_TIME + timedelta(seconds=1)),
            _point_relative_to_gate(gate, -1.0, 0.0, BASE_TIME + timedelta(seconds=2)),
        ]
        result = analyze_gate_crossing(points, gate)
        self.assertEqual(result.status, 'not_crossed')
        self.assertAlmostEqual(result.closest_approach_m, 0.5, delta=0.01)

    def test_far_away_track_never_returned_as_a_crossing(self):
        gate = _distance_g1_gate()
        points = [
            _point_relative_to_gate(gate, 500.0, 500.0, BASE_TIME),
            _point_relative_to_gate(gate, 510.0, 505.0, BASE_TIME + timedelta(seconds=1)),
        ]
        result = analyze_gate_crossing(points, gate)
        self.assertEqual(result.status, 'not_crossed')
        self.assertIsNone(result.crossing_time)
        self.assertGreater(result.closest_approach_m, 500.0)


class ExactOnLineTests(SimpleTestCase):
    """A15: exact-on-line point and multi-segment repeat, deduplicated."""

    def test_exact_on_line_point_registers_once(self):
        gate = _distance_g1_gate()
        points = [
            _point_relative_to_gate(gate, -5.0, 0.0, BASE_TIME),
            _point_relative_to_gate(gate, 0.0, 0.0, BASE_TIME + timedelta(seconds=1)),
            _point_relative_to_gate(gate, 5.0, 0.0, BASE_TIME + timedelta(seconds=2)),
        ]
        result = analyze_gate_crossing(points, gate)
        self.assertEqual(result.status, 'crossed')
        self.assertEqual(result.crossing_index, 1)
        self.assertAlmostEqual(result.alpha, 1.0, places=6)
        self.assertEqual(result.all_forward_crossing_indices, [1])

    def test_reference_index_selects_nearest_crossing_not_first(self):
        gate = _distance_g1_gate()
        points = [
            _point_relative_to_gate(gate, -1.0, 0.0, BASE_TIME),                        # 0
            _point_relative_to_gate(gate, 1.0, 0.0, BASE_TIME + timedelta(seconds=1)),  # 1: first crossing
            _point_relative_to_gate(gate, -3.0, 0.0, BASE_TIME + timedelta(seconds=2)),  # 2
            _point_relative_to_gate(gate, 7.0, 0.0, BASE_TIME + timedelta(seconds=3)),  # 3: second crossing
        ]
        # Without a reference, the first chronological crossing wins.
        default_result = analyze_gate_crossing(points, gate)
        self.assertEqual(default_result.crossing_index, 1)

        # With a reference near the second crossing, that one wins instead.
        referenced_result = analyze_gate_crossing(points, gate, reference_index=3)
        self.assertEqual(referenced_result.crossing_index, 3)
        self.assertTrue(any('closest to the reference index' in w for w in referenced_result.warnings))

    def test_multiple_forward_crossings_deduplicated_to_first(self):
        gate = _distance_g1_gate()
        points = [
            _point_relative_to_gate(gate, -5.0, 0.0, BASE_TIME),
            _point_relative_to_gate(gate, 5.0, 0.0, BASE_TIME + timedelta(seconds=1)),
            _point_relative_to_gate(gate, -3.0, 0.0, BASE_TIME + timedelta(seconds=2)),
            _point_relative_to_gate(gate, 7.0, 0.0, BASE_TIME + timedelta(seconds=3)),
        ]
        result = analyze_gate_crossing(points, gate)
        self.assertEqual(result.status, 'crossed')
        self.assertEqual(result.crossing_index, 1)
        self.assertEqual(len(result.all_forward_crossing_indices), 2)
        self.assertTrue(any('forward crossings found' in w for w in result.warnings))


class EpochFloatTimestampTests(SimpleTestCase):
    """Real flight GPS points store timestamp as a Unix epoch float (see
    flights/flight_manager.py's compact point format), not a datetime --
    the engine must work with either."""

    def test_crossing_with_float_epoch_timestamps(self):
        gate = _distance_g1_gate()
        t0 = BASE_TIME.timestamp()
        points = [
            _point_relative_to_gate(gate, -5.0, 0.0, t0, ground_speed=30.0),
            _point_relative_to_gate(gate, 5.0, 0.0, t0 + 0.25, ground_speed=30.0),
        ]
        result = analyze_gate_crossing(points, gate)
        self.assertEqual(result.status, 'crossed')
        self.assertAlmostEqual(result.alpha, 0.5, places=3)
        self.assertAlmostEqual(result.crossing_time, t0 + 0.125, places=6)
        self.assertEqual(result.confidence, 'ok')


class SampleGapConfidenceTests(SimpleTestCase):
    def test_small_gap_is_ok_confidence(self):
        gate = _distance_g1_gate()
        points = [
            _point_relative_to_gate(gate, -1.0, 0.0, BASE_TIME),
            _point_relative_to_gate(gate, 1.0, 0.0, BASE_TIME + timedelta(seconds=0.25)),
        ]
        result = analyze_gate_crossing(points, gate)
        self.assertEqual(result.confidence, 'ok')

    def test_moderate_gap_is_low_confidence(self):
        gate = _distance_g1_gate()
        points = [
            _point_relative_to_gate(gate, -1.0, 0.0, BASE_TIME),
            _point_relative_to_gate(gate, 1.0, 0.0, BASE_TIME + timedelta(seconds=0.75)),
        ]
        result = analyze_gate_crossing(points, gate)
        self.assertEqual(result.confidence, 'low_confidence_gap')

    def test_large_gap_is_ambiguous_but_still_reported(self):
        gate = _distance_g1_gate()
        points = [
            _point_relative_to_gate(gate, -1.0, 0.0, BASE_TIME),
            _point_relative_to_gate(gate, 1.0, 0.0, BASE_TIME + timedelta(seconds=1.5)),
        ]
        result = analyze_gate_crossing(points, gate)
        self.assertEqual(result.status, 'crossed')
        self.assertEqual(result.confidence, 'ambiguous_gap')
        self.assertTrue(any('do not auto-classify' in w for w in result.warnings))


class CourseWideCrossingTests(SimpleTestCase):
    def test_analyze_course_crossings_covers_every_gate(self):
        setup = _setup()
        geo = generate_distance_geometry(setup)
        g1, g5 = geo['gates']

        points = [
            _point_relative_to_gate(g1, -5.0, 0.0, BASE_TIME),
            _point_relative_to_gate(g1, 55.0, 0.0, BASE_TIME + timedelta(seconds=2)),
        ]
        results = analyze_course_crossings(points, geo['gates'])
        self.assertEqual(set(results.keys()), {g1['id'], g5['id']})
        self.assertEqual(results[g1['id']].status, 'crossed')
        self.assertEqual(results[g5['id']].status, 'crossed')

    def test_speed_curved_gate_crossing_uses_correct_local_tangent(self):
        # Confirms GateFrame derives the right forward/tangent from a
        # curved-course gate's own endpoints (not assuming straight travel).
        setup = _setup()
        geo = generate_speed_geometry(setup)
        g3 = geo['gates'][2]  # a mid-arc gate, most sensitive to a wrong tangent
        points = [
            _point_relative_to_gate(g3, -5.0, 0.0, BASE_TIME),
            _point_relative_to_gate(g3, 5.0, 0.0, BASE_TIME + timedelta(seconds=1)),
        ]
        result = analyze_gate_crossing(points, g3)
        self.assertEqual(result.status, 'crossed')
        self.assertTrue(result.within_span)
        self.assertAlmostEqual(result.lateral_offset_m, 0.0, delta=0.01)
